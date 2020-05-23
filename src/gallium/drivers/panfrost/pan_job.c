/*
 * Copyright (C) 2019 Alyssa Rosenzweig
 * Copyright (C) 2014-2017 Broadcom
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#include <assert.h>

#include "drm-uapi/panfrost_drm.h"

#include "pan_bo.h"
#include "pan_context.h"
#include "util/hash_table.h"
#include "util/ralloc.h"
#include "util/format/u_format.h"
#include "util/u_pack_color.h"
#include "util/rounding.h"
#include "pan_util.h"
#include "pandecode/decode.h"
#include "panfrost-quirks.h"

/* panfrost_bo_access is here to help us keep track of batch accesses to BOs
 * and build a proper dependency graph such that batches can be pipelined for
 * better GPU utilization.
 *
 * Each accessed BO has a corresponding entry in the ->accessed_bos hash table.
 * A BO is either being written or read at any time, that's what the type field
 * encodes.
 * When the last access is a write, the batch writing the BO might have read
 * dependencies (readers that have not been executed yet and want to read the
 * previous BO content), and when the last access is a read, all readers might
 * depend on another batch to push its results to memory. That's what the
 * readers/writers keep track off.
 * There can only be one writer at any given time, if a new batch wants to
 * write to the same BO, a dependency will be added between the new writer and
 * the old writer (at the batch level), and panfrost_bo_access->writer will be
 * updated to point to the new writer.
 */
struct panfrost_bo_access {
        uint32_t type;
        struct util_dynarray readers;
        struct panfrost_batch_fence *writer;
};

static struct panfrost_batch_fence *
panfrost_create_batch_fence(struct panfrost_batch *batch)
{
        struct panfrost_batch_fence *fence;
        ASSERTED int ret;

        fence = rzalloc(NULL, struct panfrost_batch_fence);
        assert(fence);
        pipe_reference_init(&fence->reference, 1);
        fence->ctx = batch->ctx;
        fence->batch = batch;
        ret = drmSyncobjCreate(pan_device(batch->ctx->base.screen)->fd, 0,
                               &fence->syncobj);
        assert(!ret);

        return fence;
}

static void
panfrost_free_batch_fence(struct panfrost_batch_fence *fence)
{
        drmSyncobjDestroy(pan_device(fence->ctx->base.screen)->fd,
                          fence->syncobj);
        ralloc_free(fence);
}

void
panfrost_batch_fence_unreference(struct panfrost_batch_fence *fence)
{
        if (pipe_reference(&fence->reference, NULL))
                 panfrost_free_batch_fence(fence);
}

void
panfrost_batch_fence_reference(struct panfrost_batch_fence *fence)
{
        pipe_reference(NULL, &fence->reference);
}

static struct panfrost_batch *
panfrost_create_batch(struct panfrost_context *ctx,
                      const struct pipe_framebuffer_state *key)
{
        struct panfrost_batch *batch = rzalloc(ctx, struct panfrost_batch);

        batch->ctx = ctx;

        batch->bos = _mesa_hash_table_create(batch, _mesa_hash_pointer,
                                             _mesa_key_pointer_equal);

        batch->minx = batch->miny = ~0;
        batch->maxx = batch->maxy = 0;
        batch->transient_offset = 0;

        batch->out_sync = panfrost_create_batch_fence(batch);
        util_copy_framebuffer_state(&batch->key, key);

        return batch;
}

static void
panfrost_freeze_batch(struct panfrost_batch *batch)
{
        struct panfrost_context *ctx = batch->ctx;
        struct hash_entry *entry;

        /* Remove the entry in the FBO -> batch hash table if the batch
         * matches. This way, next draws/clears targeting this FBO will trigger
         * the creation of a new batch.
         */
        entry = _mesa_hash_table_search(ctx->batches, &batch->key);
        if (entry && entry->data == batch)
                _mesa_hash_table_remove(ctx->batches, entry);

        /* If this is the bound batch, the panfrost_context parameters are
         * relevant so submitting it invalidates those parameters, but if it's
         * not bound, the context parameters are for some other batch so we
         * can't invalidate them.
         */
        if (ctx->batch == batch) {
                panfrost_invalidate_frame(ctx);
                ctx->batch = NULL;
        }
}

#ifndef NDEBUG
static bool panfrost_batch_is_frozen(struct panfrost_batch *batch)
{
        struct panfrost_context *ctx = batch->ctx;
        struct hash_entry *entry;

        entry = _mesa_hash_table_search(ctx->batches, &batch->key);
        if (entry && entry->data == batch)
                return false;

        if (ctx->batch == batch)
                return false;

        return true;
}
#endif

static void
panfrost_free_batch(struct panfrost_batch *batch)
{
        if (!batch)
                return;

        assert(panfrost_batch_is_frozen(batch));

        hash_table_foreach(batch->bos, entry)
                panfrost_bo_unreference((struct panfrost_bo *)entry->key);

        util_dynarray_foreach(&batch->dependencies,
                              struct panfrost_batch_fence *, dep) {
                panfrost_batch_fence_unreference(*dep);
        }

        /* The out_sync fence lifetime is different from the the batch one
         * since other batches might want to wait on a fence of already
         * submitted/signaled batch. All we need to do here is make sure the
         * fence does not point to an invalid batch, which the core will
         * interpret as 'batch is already submitted'.
         */
        batch->out_sync->batch = NULL;
        panfrost_batch_fence_unreference(batch->out_sync);

        util_unreference_framebuffer_state(&batch->key);
        ralloc_free(batch);
}

#ifndef NDEBUG
static bool
panfrost_dep_graph_contains_batch(struct panfrost_batch *root,
                                  struct panfrost_batch *batch)
{
        if (!root)
                return false;

        util_dynarray_foreach(&root->dependencies,
                              struct panfrost_batch_fence *, dep) {
                if ((*dep)->batch == batch ||
                    panfrost_dep_graph_contains_batch((*dep)->batch, batch))
                        return true;
        }

        return false;
}
#endif

static void
panfrost_batch_add_dep(struct panfrost_batch *batch,
                       struct panfrost_batch_fence *newdep)
{
        if (batch == newdep->batch)
                return;

        /* We might want to turn ->dependencies into a set if the number of
         * deps turns out to be big enough to make this 'is dep already there'
         * search inefficient.
         */
        util_dynarray_foreach(&batch->dependencies,
                              struct panfrost_batch_fence *, dep) {
                if (*dep == newdep)
                        return;
        }

        /* Make sure the dependency graph is acyclic. */
        assert(!panfrost_dep_graph_contains_batch(newdep->batch, batch));

        panfrost_batch_fence_reference(newdep);
        util_dynarray_append(&batch->dependencies,
                             struct panfrost_batch_fence *, newdep);

        /* We now have a batch depending on us, let's make sure new draw/clear
         * calls targeting the same FBO use a new batch object.
         */
        if (newdep->batch)
                panfrost_freeze_batch(newdep->batch);
}

static struct panfrost_batch *
panfrost_get_batch(struct panfrost_context *ctx,
                   const struct pipe_framebuffer_state *key)
{
        /* Lookup the job first */
        struct hash_entry *entry = _mesa_hash_table_search(ctx->batches, key);

        if (entry)
                return entry->data;

        /* Otherwise, let's create a job */

        struct panfrost_batch *batch = panfrost_create_batch(ctx, key);

        /* Save the created job */
        _mesa_hash_table_insert(ctx->batches, &batch->key, batch);

        return batch;
}

/* Get the job corresponding to the FBO we're currently rendering into */

struct panfrost_batch *
panfrost_get_batch_for_fbo(struct panfrost_context *ctx)
{
        /* If we're wallpapering, we special case to workaround
         * u_blitter abuse */

        if (ctx->wallpaper_batch)
                return ctx->wallpaper_batch;

        /* If we already began rendering, use that */

        if (ctx->batch) {
                assert(util_framebuffer_state_equal(&ctx->batch->key,
                                                    &ctx->pipe_framebuffer));
                return ctx->batch;
        }

        /* If not, look up the job */
        struct panfrost_batch *batch = panfrost_get_batch(ctx,
                                                          &ctx->pipe_framebuffer);

        /* Set this job as the current FBO job. Will be reset when updating the
         * FB state and when submitting or releasing a job.
         */
        ctx->batch = batch;
        return batch;
}

struct panfrost_batch *
panfrost_get_fresh_batch_for_fbo(struct panfrost_context *ctx)
{
        struct panfrost_batch *batch;

        batch = panfrost_get_batch(ctx, &ctx->pipe_framebuffer);

        /* The batch has no draw/clear queued, let's return it directly.
         * Note that it's perfectly fine to re-use a batch with an
         * existing clear, we'll just update it with the new clear request.
         */
        if (!batch->first_job)
                return batch;

        /* Otherwise, we need to freeze the existing one and instantiate a new
         * one.
         */
        panfrost_freeze_batch(batch);
        return panfrost_get_batch(ctx, &ctx->pipe_framebuffer);
}

static bool
panfrost_batch_fence_is_signaled(struct panfrost_batch_fence *fence)
{
        if (fence->signaled)
                return true;

        /* Batch has not been submitted yet. */
        if (fence->batch)
                return false;

        int ret = drmSyncobjWait(pan_device(fence->ctx->base.screen)->fd,
                                 &fence->syncobj, 1, 0, 0, NULL);

        /* Cache whether the fence was signaled */
        fence->signaled = ret >= 0;
        return fence->signaled;
}

static void
panfrost_bo_access_gc_fences(struct panfrost_context *ctx,
                             struct panfrost_bo_access *access,
			     const struct panfrost_bo *bo)
{
        if (access->writer && panfrost_batch_fence_is_signaled(access->writer)) {
                panfrost_batch_fence_unreference(access->writer);
                access->writer = NULL;
        }

        struct panfrost_batch_fence **readers_array = util_dynarray_begin(&access->readers);
        struct panfrost_batch_fence **new_readers = readers_array;

        util_dynarray_foreach(&access->readers, struct panfrost_batch_fence *,
                              reader) {
                if (!(*reader))
                        continue;

                if (panfrost_batch_fence_is_signaled(*reader)) {
                        panfrost_batch_fence_unreference(*reader);
                        *reader = NULL;
                } else {
                        /* Build a new array of only unsignaled fences in-place */
                        *(new_readers++) = *reader;
                }
        }

        if (!util_dynarray_resize(&access->readers, struct panfrost_batch_fence *,
                                  new_readers - readers_array) &&
            new_readers != readers_array)
                unreachable("Invalid dynarray access->readers");
}

/* Collect signaled fences to keep the kernel-side syncobj-map small. The
 * idea is to collect those signaled fences at the end of each flush_all
 * call. This function is likely to collect only fences from previous
 * batch flushes not the one that have just have just been submitted and
 * are probably still in flight when we trigger the garbage collection.
 * Anyway, we need to do this garbage collection at some point if we don't
 * want the BO access map to keep invalid entries around and retain
 * syncobjs forever.
 */
static void
panfrost_gc_fences(struct panfrost_context *ctx)
{
        hash_table_foreach(ctx->accessed_bos, entry) {
                struct panfrost_bo_access *access = entry->data;

                assert(access);
                panfrost_bo_access_gc_fences(ctx, access, entry->key);
                if (!util_dynarray_num_elements(&access->readers,
                                                struct panfrost_batch_fence *) &&
                    !access->writer) {
                        ralloc_free(access);
                        _mesa_hash_table_remove(ctx->accessed_bos, entry);
                }
        }
}

#ifndef NDEBUG
static bool
panfrost_batch_in_readers(struct panfrost_batch *batch,
                          struct panfrost_bo_access *access)
{
        util_dynarray_foreach(&access->readers, struct panfrost_batch_fence *,
                              reader) {
                if (*reader && (*reader)->batch == batch)
                        return true;
        }

        return false;
}
#endif

static void
panfrost_batch_update_bo_access(struct panfrost_batch *batch,
                                struct panfrost_bo *bo, uint32_t access_type,
                                bool already_accessed)
{
        struct panfrost_context *ctx = batch->ctx;
        struct panfrost_bo_access *access;
        uint32_t old_access_type;
        struct hash_entry *entry;

        assert(access_type == PAN_BO_ACCESS_WRITE ||
               access_type == PAN_BO_ACCESS_READ);

        entry = _mesa_hash_table_search(ctx->accessed_bos, bo);
        access = entry ? entry->data : NULL;
        if (access) {
                old_access_type = access->type;
        } else {
                access = rzalloc(ctx, struct panfrost_bo_access);
                util_dynarray_init(&access->readers, access);
                _mesa_hash_table_insert(ctx->accessed_bos, bo, access);
                /* We are the first to access this BO, let's initialize
                 * old_access_type to our own access type in that case.
                 */
                old_access_type = access_type;
                access->type = access_type;
        }

        assert(access);

        if (access_type == PAN_BO_ACCESS_WRITE &&
            old_access_type == PAN_BO_ACCESS_READ) {
                /* Previous access was a read and we want to write this BO.
                 * We first need to add explicit deps between our batch and
                 * the previous readers.
                 */
                util_dynarray_foreach(&access->readers,
                                      struct panfrost_batch_fence *, reader) {
                        /* We were already reading the BO, no need to add a dep
                         * on ourself (the acyclic check would complain about
                         * that).
                         */
                        if (!(*reader) || (*reader)->batch == batch)
                                continue;

                        panfrost_batch_add_dep(batch, *reader);
                }
                panfrost_batch_fence_reference(batch->out_sync);

                /* We now are the new writer. */
                access->writer = batch->out_sync;
                access->type = access_type;

                /* Release the previous readers and reset the readers array. */
                util_dynarray_foreach(&access->readers,
                                      struct panfrost_batch_fence *,
                                      reader) {
                        if (!*reader)
                                continue;
                        panfrost_batch_fence_unreference(*reader);
                }

                util_dynarray_clear(&access->readers);
        } else if (access_type == PAN_BO_ACCESS_WRITE &&
                   old_access_type == PAN_BO_ACCESS_WRITE) {
                /* Previous access was a write and we want to write this BO.
                 * First check if we were the previous writer, in that case
                 * there's nothing to do. Otherwise we need to add a
                 * dependency between the new writer and the old one.
                 */
		if (access->writer != batch->out_sync) {
                        if (access->writer) {
                                panfrost_batch_add_dep(batch, access->writer);
                                panfrost_batch_fence_unreference(access->writer);
                        }
                        panfrost_batch_fence_reference(batch->out_sync);
                        access->writer = batch->out_sync;
                }
        } else if (access_type == PAN_BO_ACCESS_READ &&
                   old_access_type == PAN_BO_ACCESS_WRITE) {
                /* Previous access was a write and we want to read this BO.
                 * First check if we were the previous writer, in that case
                 * we want to keep the access type unchanged, as a write is
                 * more constraining than a read.
                 */
                if (access->writer != batch->out_sync) {
                        /* Add a dependency on the previous writer. */
                        panfrost_batch_add_dep(batch, access->writer);

                        /* The previous access was a write, there's no reason
                         * to have entries in the readers array.
                         */
                        assert(!util_dynarray_num_elements(&access->readers,
                                                           struct panfrost_batch_fence *));

                        /* Add ourselves to the readers array. */
                        panfrost_batch_fence_reference(batch->out_sync);
                        util_dynarray_append(&access->readers,
                                             struct panfrost_batch_fence *,
                                             batch->out_sync);
                        access->type = PAN_BO_ACCESS_READ;
                }
        } else {
                /* We already accessed this BO before, so we should already be
                 * in the reader array.
                 */
                if (already_accessed) {
                        assert(panfrost_batch_in_readers(batch, access));
                        return;
                }

                /* Previous access was a read and we want to read this BO.
                 * Add ourselves to the readers array and add a dependency on
                 * the previous writer if any.
                 */
                panfrost_batch_fence_reference(batch->out_sync);
                util_dynarray_append(&access->readers,
                                     struct panfrost_batch_fence *,
                                     batch->out_sync);

                if (access->writer)
                        panfrost_batch_add_dep(batch, access->writer);
        }
}

void
panfrost_batch_add_bo(struct panfrost_batch *batch, struct panfrost_bo *bo,
                      uint32_t flags)
{
        if (!bo)
                return;

        struct hash_entry *entry;
        uint32_t old_flags = 0;

        entry = _mesa_hash_table_search(batch->bos, bo);
        if (!entry) {
                entry = _mesa_hash_table_insert(batch->bos, bo,
                                                (void *)(uintptr_t)flags);
                panfrost_bo_reference(bo);
	} else {
                old_flags = (uintptr_t)entry->data;

                /* All batches have to agree on the shared flag. */
                assert((old_flags & PAN_BO_ACCESS_SHARED) ==
                       (flags & PAN_BO_ACCESS_SHARED));
        }

        assert(entry);

        if (old_flags == flags)
                return;

        flags |= old_flags;
        entry->data = (void *)(uintptr_t)flags;

        /* If this is not a shared BO, we don't really care about dependency
         * tracking.
         */
        if (!(flags & PAN_BO_ACCESS_SHARED))
                return;

        /* All dependencies should have been flushed before we execute the
         * wallpaper draw, so it should be harmless to skip the
         * update_bo_access() call.
         */
        if (batch == batch->ctx->wallpaper_batch)
                return;

        /* Only pass R/W flags to the dep tracking logic. */
        assert(flags & PAN_BO_ACCESS_RW);
        flags = (flags & PAN_BO_ACCESS_WRITE) ?
                PAN_BO_ACCESS_WRITE : PAN_BO_ACCESS_READ;
        panfrost_batch_update_bo_access(batch, bo, flags, old_flags != 0);
}

static void
panfrost_batch_add_resource_bos(struct panfrost_batch *batch,
                                struct panfrost_resource *rsrc,
                                uint32_t flags)
{
        panfrost_batch_add_bo(batch, rsrc->bo, flags);

        for (unsigned i = 0; i < MAX_MIP_LEVELS; i++)
                if (rsrc->slices[i].checksum_bo)
                        panfrost_batch_add_bo(batch, rsrc->slices[i].checksum_bo, flags);
}

void panfrost_batch_add_fbo_bos(struct panfrost_batch *batch)
{
        uint32_t flags = PAN_BO_ACCESS_SHARED | PAN_BO_ACCESS_WRITE |
                         PAN_BO_ACCESS_VERTEX_TILER |
                         PAN_BO_ACCESS_FRAGMENT;

        for (unsigned i = 0; i < batch->key.nr_cbufs; ++i) {
                struct panfrost_resource *rsrc = pan_resource(batch->key.cbufs[i]->texture);
                panfrost_batch_add_resource_bos(batch, rsrc, flags);
        }

        if (batch->key.zsbuf) {
                struct panfrost_resource *rsrc = pan_resource(batch->key.zsbuf->texture);
                panfrost_batch_add_resource_bos(batch, rsrc, flags);
        }
}

struct panfrost_bo *
panfrost_batch_create_bo(struct panfrost_batch *batch, size_t size,
                         uint32_t create_flags, uint32_t access_flags)
{
        struct panfrost_bo *bo;

        bo = pan_bo_create(pan_device(batch->ctx->base.screen), size,
                                create_flags);
        panfrost_batch_add_bo(batch, bo, access_flags);

        /* panfrost_batch_add_bo() has retained a reference and
         * pan_bo_create() initialize the refcnt to 1, so let's
         * unreference the BO here so it gets released when the batch is
         * destroyed (unless it's retained by someone else in the meantime).
         */
        panfrost_bo_unreference(bo);
        return bo;
}

/* Returns the polygon list's GPU address if available, or otherwise allocates
 * the polygon list.  It's perfectly fast to use allocate/free BO directly,
 * since we'll hit the BO cache and this is one-per-batch anyway. */

mali_ptr
panfrost_batch_get_polygon_list(struct panfrost_batch *batch, unsigned size)
{
        if (batch->polygon_list) {
                assert(batch->polygon_list->size >= size);
        } else {
                /* Create the BO as invisible, as there's no reason to map */
                size = util_next_power_of_two(size);

                batch->polygon_list = panfrost_batch_create_bo(batch, size,
                                                               PAN_BO_INVISIBLE,
                                                               PAN_BO_ACCESS_PRIVATE |
                                                               PAN_BO_ACCESS_RW |
                                                               PAN_BO_ACCESS_VERTEX_TILER |
                                                               PAN_BO_ACCESS_FRAGMENT);
        }

        return batch->polygon_list->gpu;
}

struct panfrost_bo *
panfrost_batch_get_scratchpad(struct panfrost_batch *batch,
                unsigned shift,
                unsigned thread_tls_alloc,
                unsigned core_count)
{
        unsigned size = panfrost_get_total_stack_size(shift,
                        thread_tls_alloc,
                        core_count);

        if (batch->scratchpad) {
                assert(batch->scratchpad->size >= size);
        } else {
                batch->scratchpad = panfrost_batch_create_bo(batch, size,
                                             PAN_BO_INVISIBLE,
                                             PAN_BO_ACCESS_PRIVATE |
                                             PAN_BO_ACCESS_RW |
                                             PAN_BO_ACCESS_VERTEX_TILER |
                                             PAN_BO_ACCESS_FRAGMENT);
        }

        return batch->scratchpad;
}

struct panfrost_bo *
panfrost_batch_get_shared_memory(struct panfrost_batch *batch,
                unsigned size,
                unsigned workgroup_count)
{
        if (batch->shared_memory) {
                assert(batch->shared_memory->size >= size);
        } else {
                batch->shared_memory = panfrost_batch_create_bo(batch, size,
                                             PAN_BO_INVISIBLE,
                                             PAN_BO_ACCESS_PRIVATE |
                                             PAN_BO_ACCESS_RW |
                                             PAN_BO_ACCESS_VERTEX_TILER);
        }

        return batch->shared_memory;
}

struct panfrost_bo *
panfrost_batch_get_tiler_heap(struct panfrost_batch *batch)
{
        if (batch->tiler_heap)
                return batch->tiler_heap;

        batch->tiler_heap = panfrost_batch_create_bo(batch, 4096 * 4096,
                                                     PAN_BO_INVISIBLE |
                                                     PAN_BO_GROWABLE,
                                                     PAN_BO_ACCESS_PRIVATE |
                                                     PAN_BO_ACCESS_RW |
                                                     PAN_BO_ACCESS_VERTEX_TILER |
                                                     PAN_BO_ACCESS_FRAGMENT);
        assert(batch->tiler_heap);
        return batch->tiler_heap;
}

mali_ptr
panfrost_batch_get_tiler_meta(struct panfrost_batch *batch, unsigned vertex_count)
{
        if (!vertex_count)
                return 0;

        if (batch->tiler_meta)
                return batch->tiler_meta;

        struct panfrost_bo *tiler_heap;
        tiler_heap = panfrost_batch_get_tiler_heap(batch);

        struct bifrost_tiler_heap_meta tiler_heap_meta = {
            .heap_size = tiler_heap->size,
            .tiler_heap_start = tiler_heap->gpu,
            .tiler_heap_free = tiler_heap->gpu,
            .tiler_heap_end = tiler_heap->gpu + tiler_heap->size,
            .unk1 = 0x1,
            .unk7e007e = 0x7e007e,
        };

        struct bifrost_tiler_meta tiler_meta = {
            .hierarchy_mask = 0x28,
            .flags = 0x0,
            .width = MALI_POSITIVE(batch->key.width),
            .height = MALI_POSITIVE(batch->key.height),
            .tiler_heap_meta = panfrost_upload_transient(batch, &tiler_heap_meta, sizeof(tiler_heap_meta)),
        };

        batch->tiler_meta = panfrost_upload_transient(batch, &tiler_meta, sizeof(tiler_meta));
        return batch->tiler_meta;
}

struct panfrost_bo *
panfrost_batch_get_tiler_dummy(struct panfrost_batch *batch)
{
        struct panfrost_device *dev = pan_device(batch->ctx->base.screen);

        uint32_t create_flags = 0;

        if (batch->tiler_dummy)
                return batch->tiler_dummy;

        if (!(dev->quirks & MIDGARD_NO_HIER_TILING))
                create_flags = PAN_BO_INVISIBLE;

        batch->tiler_dummy = panfrost_batch_create_bo(batch, 4096,
                                                      create_flags,
                                                      PAN_BO_ACCESS_PRIVATE |
                                                      PAN_BO_ACCESS_RW |
                                                      PAN_BO_ACCESS_VERTEX_TILER |
                                                      PAN_BO_ACCESS_FRAGMENT);
        assert(batch->tiler_dummy);
        return batch->tiler_dummy;
}

static void
panfrost_batch_draw_wallpaper(struct panfrost_batch *batch)
{
        /* Color 0 is cleared, no need to draw the wallpaper.
         * TODO: MRT wallpapers.
         */
        if (batch->clear & PIPE_CLEAR_COLOR0)
                return;

        /* Nothing to reload? TODO: MRT wallpapers */
        if (batch->key.cbufs[0] == NULL)
                return;

        /* No draw calls, and no clear on the depth/stencil bufs.
         * Drawing the wallpaper would be useless.
         */
        if (!batch->tiler_dep &&
            !(batch->clear & PIPE_CLEAR_DEPTHSTENCIL))
                return;

        /* Check if the buffer has any content on it worth preserving */

        struct pipe_surface *surf = batch->key.cbufs[0];
        struct panfrost_resource *rsrc = pan_resource(surf->texture);
        unsigned level = surf->u.tex.level;

        if (!rsrc->slices[level].initialized)
                return;

        batch->ctx->wallpaper_batch = batch;

        /* Clamp the rendering area to the damage extent. The
         * KHR_partial_update() spec states that trying to render outside of
         * the damage region is "undefined behavior", so we should be safe.
         */
        unsigned damage_width = (rsrc->damage.extent.maxx - rsrc->damage.extent.minx);
        unsigned damage_height = (rsrc->damage.extent.maxy - rsrc->damage.extent.miny);

        if (damage_width && damage_height) {
                panfrost_batch_intersection_scissor(batch,
                                                    rsrc->damage.extent.minx,
                                                    rsrc->damage.extent.miny,
                                                    rsrc->damage.extent.maxx,
                                                    rsrc->damage.extent.maxy);
        }

        /* FIXME: Looks like aligning on a tile is not enough, but
         * aligning on twice the tile size seems to works. We don't
         * know exactly what happens here but this deserves extra
         * investigation to figure it out.
         */
        batch->minx = batch->minx & ~((MALI_TILE_LENGTH * 2) - 1);
        batch->miny = batch->miny & ~((MALI_TILE_LENGTH * 2) - 1);
        batch->maxx = MIN2(ALIGN_POT(batch->maxx, MALI_TILE_LENGTH * 2),
                           rsrc->base.width0);
        batch->maxy = MIN2(ALIGN_POT(batch->maxy, MALI_TILE_LENGTH * 2),
                           rsrc->base.height0);

        struct pipe_scissor_state damage;
        struct pipe_box rects[4];

        /* Clamp the damage box to the rendering area. */
        damage.minx = MAX2(batch->minx, rsrc->damage.biggest_rect.x);
        damage.miny = MAX2(batch->miny, rsrc->damage.biggest_rect.y);
        damage.maxx = MIN2(batch->maxx,
                           rsrc->damage.biggest_rect.x +
                           rsrc->damage.biggest_rect.width);
        damage.maxx = MAX2(damage.maxx, damage.minx);
        damage.maxy = MIN2(batch->maxy,
                           rsrc->damage.biggest_rect.y +
                           rsrc->damage.biggest_rect.height);
        damage.maxy = MAX2(damage.maxy, damage.miny);

        /* One damage rectangle means we can end up with at most 4 reload
         * regions:
         * 1: left region, only exists if damage.x > 0
         * 2: right region, only exists if damage.x + damage.width < fb->width
         * 3: top region, only exists if damage.y > 0. The intersection with
         *    the left and right regions are dropped
         * 4: bottom region, only exists if damage.y + damage.height < fb->height.
         *    The intersection with the left and right regions are dropped
         *
         *                    ____________________________
         *                    |       |     3     |      |
         *                    |       |___________|      |
         *                    |       |   damage  |      |
         *                    |   1   |    rect   |   2  |
         *                    |       |___________|      |
         *                    |       |     4     |      |
         *                    |_______|___________|______|
         */
        u_box_2d(batch->minx, batch->miny, damage.minx - batch->minx,
                 batch->maxy - batch->miny, &rects[0]);
        u_box_2d(damage.maxx, batch->miny, batch->maxx - damage.maxx,
                 batch->maxy - batch->miny, &rects[1]);
        u_box_2d(damage.minx, batch->miny, damage.maxx - damage.minx,
                 damage.miny - batch->miny, &rects[2]);
        u_box_2d(damage.minx, damage.maxy, damage.maxx - damage.minx,
                 batch->maxy - damage.maxy, &rects[3]);

        for (unsigned i = 0; i < 4; i++) {
                /* Width and height are always >= 0 even if width is declared as a
                 * signed integer: u_box_2d() helper takes unsigned args and
                 * panfrost_set_damage_region() is taking care of clamping
                 * negative values.
                 */
                if (!rects[i].width || !rects[i].height)
                        continue;

                /* Blit the wallpaper in */
                panfrost_blit_wallpaper(batch->ctx, &rects[i]);
        }
        batch->ctx->wallpaper_batch = NULL;
}

static int
panfrost_batch_submit_ioctl(struct panfrost_batch *batch,
                            mali_ptr first_job_desc,
                            uint32_t reqs)
{
        struct panfrost_context *ctx = batch->ctx;
        struct pipe_context *gallium = (struct pipe_context *) ctx;
        struct panfrost_device *dev = pan_device(gallium->screen);
        struct drm_panfrost_submit submit = {0,};
        uint32_t *bo_handles, *in_syncs = NULL;
        bool is_fragment_shader;
        int ret;

        is_fragment_shader = (reqs & PANFROST_JD_REQ_FS) && batch->first_job;
        if (is_fragment_shader)
                submit.in_sync_count = 1;
        else
                submit.in_sync_count = util_dynarray_num_elements(&batch->dependencies,
                                                                  struct panfrost_batch_fence *);

        if (submit.in_sync_count) {
                in_syncs = calloc(submit.in_sync_count, sizeof(*in_syncs));
                assert(in_syncs);
        }

        /* The fragment job always depends on the vertex/tiler job if there's
         * one
         */
        if (is_fragment_shader) {
                in_syncs[0] = batch->out_sync->syncobj;
        } else {
                unsigned int i = 0;

                util_dynarray_foreach(&batch->dependencies,
                                      struct panfrost_batch_fence *, dep)
                        in_syncs[i++] = (*dep)->syncobj;
        }

        submit.in_syncs = (uintptr_t)in_syncs;
        submit.out_sync = batch->out_sync->syncobj;
        submit.jc = first_job_desc;
        submit.requirements = reqs;

        bo_handles = calloc(batch->bos->entries, sizeof(*bo_handles));
        assert(bo_handles);

        hash_table_foreach(batch->bos, entry) {
                struct panfrost_bo *bo = (struct panfrost_bo *)entry->key;
                uint32_t flags = (uintptr_t)entry->data;

                assert(bo->gem_handle > 0);
                bo_handles[submit.bo_handle_count++] = bo->gem_handle;

                /* Update the BO access flags so that panfrost_bo_wait() knows
                 * about all pending accesses.
                 * We only keep the READ/WRITE info since this is all the BO
                 * wait logic cares about.
                 * We also preserve existing flags as this batch might not
                 * be the first one to access the BO.
                 */
                bo->gpu_access |= flags & (PAN_BO_ACCESS_RW);
        }

        submit.bo_handles = (u64) (uintptr_t) bo_handles;
        ret = drmIoctl(dev->fd, DRM_IOCTL_PANFROST_SUBMIT, &submit);
        free(bo_handles);
        free(in_syncs);

        if (ret) {
                DBG("Error submitting: %m\n");
                return errno;
        }

        /* Trace the job if we're doing that */
        if (pan_debug & (PAN_DBG_TRACE | PAN_DBG_SYNC)) {
                /* Wait so we can get errors reported back */
                drmSyncobjWait(dev->fd, &batch->out_sync->syncobj, 1,
                               INT64_MAX, 0, NULL);

                /* Trace gets priority over sync */
                bool minimal = !(pan_debug & PAN_DBG_TRACE);
                pandecode_jc(submit.jc, dev->quirks & IS_BIFROST, dev->gpu_id, minimal);
        }

        return 0;
}

static int
panfrost_batch_submit_jobs(struct panfrost_batch *batch)
{
        bool has_draws = batch->first_job;
        int ret = 0;

        if (has_draws) {
                ret = panfrost_batch_submit_ioctl(batch, batch->first_job, 0);
                assert(!ret);
        }

        if (batch->tiler_dep || batch->clear) {
                mali_ptr fragjob = panfrost_fragment_job(batch, has_draws);
                ret = panfrost_batch_submit_ioctl(batch, fragjob, PANFROST_JD_REQ_FS);
                assert(!ret);
        }

        return ret;
}

static void
panfrost_batch_submit(struct panfrost_batch *batch)
{
        assert(batch);

        /* Submit the dependencies first. */
        util_dynarray_foreach(&batch->dependencies,
                              struct panfrost_batch_fence *, dep) {
                if ((*dep)->batch)
                        panfrost_batch_submit((*dep)->batch);
        }

        int ret;

        /* Nothing to do! */
        if (!batch->first_job && !batch->clear) {
                /* Mark the fence as signaled so the fence logic does not try
                 * to wait on it.
                 */
                batch->out_sync->signaled = true;
                goto out;
        }

        panfrost_batch_draw_wallpaper(batch);

        /* Now that all draws are in, we can finally prepare the
         * FBD for the batch */

        if (batch->framebuffer.gpu && batch->first_job) {
                struct panfrost_context *ctx = batch->ctx;
                struct pipe_context *gallium = (struct pipe_context *) ctx;
                struct panfrost_device *dev = pan_device(gallium->screen);

                if (dev->quirks & MIDGARD_SFBD)
                        panfrost_attach_sfbd(batch, ~0);
                else
                        panfrost_attach_mfbd(batch, ~0);
        }

        panfrost_scoreboard_initialize_tiler(batch);

        ret = panfrost_batch_submit_jobs(batch);

        if (ret)
                DBG("panfrost_batch_submit failed: %d\n", ret);

        /* We must reset the damage info of our render targets here even
         * though a damage reset normally happens when the DRI layer swaps
         * buffers. That's because there can be implicit flushes the GL
         * app is not aware of, and those might impact the damage region: if
         * part of the damaged portion is drawn during those implicit flushes,
         * you have to reload those areas before next draws are pushed, and
         * since the driver can't easily know what's been modified by the draws
         * it flushed, the easiest solution is to reload everything.
         */
        for (unsigned i = 0; i < batch->key.nr_cbufs; i++) {
                struct panfrost_resource *res;

                if (!batch->key.cbufs[i])
                        continue;

                res = pan_resource(batch->key.cbufs[i]->texture);
                panfrost_resource_reset_damage(res);
        }

out:
        panfrost_freeze_batch(batch);
        panfrost_free_batch(batch);
}

void
panfrost_flush_all_batches(struct panfrost_context *ctx, bool wait)
{
        struct util_dynarray fences, syncobjs;

        if (wait) {
                util_dynarray_init(&fences, NULL);
                util_dynarray_init(&syncobjs, NULL);
        }

        hash_table_foreach(ctx->batches, hentry) {
                struct panfrost_batch *batch = hentry->data;

                assert(batch);

                if (wait) {
                        panfrost_batch_fence_reference(batch->out_sync);
                        util_dynarray_append(&fences, struct panfrost_batch_fence *,
                                             batch->out_sync);
                        util_dynarray_append(&syncobjs, uint32_t,
                                             batch->out_sync->syncobj);
                }

                panfrost_batch_submit(batch);
        }

        assert(!ctx->batches->entries);

        /* Collect batch fences before returning */
        panfrost_gc_fences(ctx);

        if (!wait)
                return;

        drmSyncobjWait(pan_device(ctx->base.screen)->fd,
                       util_dynarray_begin(&syncobjs),
                       util_dynarray_num_elements(&syncobjs, uint32_t),
                       INT64_MAX, DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL, NULL);

        util_dynarray_foreach(&fences, struct panfrost_batch_fence *, fence)
                panfrost_batch_fence_unreference(*fence);

        util_dynarray_fini(&fences);
        util_dynarray_fini(&syncobjs);
}

bool
panfrost_pending_batches_access_bo(struct panfrost_context *ctx,
                                   const struct panfrost_bo *bo)
{
        struct panfrost_bo_access *access;
        struct hash_entry *hentry;

        hentry = _mesa_hash_table_search(ctx->accessed_bos, bo);
        access = hentry ? hentry->data : NULL;
        if (!access)
                return false;

        if (access->writer && access->writer->batch)
                return true;

        util_dynarray_foreach(&access->readers, struct panfrost_batch_fence *,
                              reader) {
                if (*reader && (*reader)->batch)
                        return true;
        }

        return false;
}

void
panfrost_flush_batches_accessing_bo(struct panfrost_context *ctx,
                                    struct panfrost_bo *bo,
                                    uint32_t access_type)
{
        struct panfrost_bo_access *access;
        struct hash_entry *hentry;

        /* It doesn't make any to flush only the readers. */
        assert(access_type == PAN_BO_ACCESS_WRITE ||
               access_type == PAN_BO_ACCESS_RW);

        hentry = _mesa_hash_table_search(ctx->accessed_bos, bo);
        access = hentry ? hentry->data : NULL;
        if (!access)
                return;

        if (access_type & PAN_BO_ACCESS_WRITE && access->writer &&
            access->writer->batch)
                panfrost_batch_submit(access->writer->batch);

        if (!(access_type & PAN_BO_ACCESS_READ))
                return;

        util_dynarray_foreach(&access->readers, struct panfrost_batch_fence *,
                              reader) {
                if (*reader && (*reader)->batch)
                        panfrost_batch_submit((*reader)->batch);
        }
}

void
panfrost_batch_set_requirements(struct panfrost_batch *batch)
{
        struct panfrost_context *ctx = batch->ctx;

        if (ctx->rasterizer && ctx->rasterizer->base.multisample)
                batch->requirements |= PAN_REQ_MSAA;

        if (ctx->depth_stencil && ctx->depth_stencil->depth.writemask)
                batch->requirements |= PAN_REQ_DEPTH_WRITE;
}

void
panfrost_batch_adjust_stack_size(struct panfrost_batch *batch)
{
        struct panfrost_context *ctx = batch->ctx;

        for (unsigned i = 0; i < PIPE_SHADER_TYPES; ++i) {
                struct panfrost_shader_state *ss;

                ss = panfrost_get_shader_state(ctx, i);
                if (!ss)
                        continue;

                batch->stack_size = MAX2(batch->stack_size, ss->stack_size);
        }
}

/* Helper to smear a 32-bit color across 128-bit components */

static void
pan_pack_color_32(uint32_t *packed, uint32_t v)
{
        for (unsigned i = 0; i < 4; ++i)
                packed[i] = v;
}

static void
pan_pack_color_64(uint32_t *packed, uint32_t lo, uint32_t hi)
{
        for (unsigned i = 0; i < 4; i += 2) {
                packed[i + 0] = lo;
                packed[i + 1] = hi;
        }
}

static void
pan_pack_color(uint32_t *packed, const union pipe_color_union *color, enum pipe_format format)
{
        /* Alpha magicked to 1.0 if there is no alpha */

        bool has_alpha = util_format_has_alpha(format);
        float clear_alpha = has_alpha ? color->f[3] : 1.0f;

        /* Packed color depends on the framebuffer format */

        const struct util_format_description *desc =
                util_format_description(format);

        if (util_format_is_rgba8_variant(desc)) {
                pan_pack_color_32(packed,
                                  ((uint32_t) float_to_ubyte(clear_alpha) << 24) |
                                  ((uint32_t) float_to_ubyte(color->f[2]) << 16) |
                                  ((uint32_t) float_to_ubyte(color->f[1]) <<  8) |
                                  ((uint32_t) float_to_ubyte(color->f[0]) <<  0));
        } else if (format == PIPE_FORMAT_B5G6R5_UNORM) {
                /* First, we convert the components to R5, G6, B5 separately */
                unsigned r5 = _mesa_roundevenf(CLAMP(color->f[0], 0.0, 1.0) * 31.0);
                unsigned g6 = _mesa_roundevenf(CLAMP(color->f[1], 0.0, 1.0) * 63.0);
                unsigned b5 = _mesa_roundevenf(CLAMP(color->f[2], 0.0, 1.0) * 31.0);

                /* Then we pack into a sparse u32. TODO: Why these shifts? */
                pan_pack_color_32(packed, (b5 << 25) | (g6 << 14) | (r5 << 5));
        } else if (format == PIPE_FORMAT_B4G4R4A4_UNORM) {
                /* Convert to 4-bits */
                unsigned r4 = _mesa_roundevenf(CLAMP(color->f[0], 0.0, 1.0) * 15.0);
                unsigned g4 = _mesa_roundevenf(CLAMP(color->f[1], 0.0, 1.0) * 15.0);
                unsigned b4 = _mesa_roundevenf(CLAMP(color->f[2], 0.0, 1.0) * 15.0);
                unsigned a4 = _mesa_roundevenf(CLAMP(clear_alpha, 0.0, 1.0) * 15.0);

                /* Pack on *byte* intervals */
                pan_pack_color_32(packed, (a4 << 28) | (b4 << 20) | (g4 << 12) | (r4 << 4));
        } else if (format == PIPE_FORMAT_B5G5R5A1_UNORM) {
                /* Scale as expected but shift oddly */
                unsigned r5 = _mesa_roundevenf(CLAMP(color->f[0], 0.0, 1.0) * 31.0);
                unsigned g5 = _mesa_roundevenf(CLAMP(color->f[1], 0.0, 1.0) * 31.0);
                unsigned b5 = _mesa_roundevenf(CLAMP(color->f[2], 0.0, 1.0) * 31.0);
                unsigned a1 = _mesa_roundevenf(CLAMP(clear_alpha, 0.0, 1.0) * 1.0);

                pan_pack_color_32(packed, (a1 << 31) | (b5 << 25) | (g5 << 15) | (r5 << 5));
        } else {
                /* Try Gallium's generic default path. Doesn't work for all
                 * formats but it's a good guess. */

                union util_color out;

                if (util_format_is_pure_integer(format)) {
                        memcpy(out.ui, color->ui, 16);
                } else {
                        util_pack_color(color->f, format, &out);
                }

                unsigned size = util_format_get_blocksize(format);

                if (size == 1) {
                        unsigned b = out.ui[0];
                        unsigned s = b | (b << 8);
                        pan_pack_color_32(packed, s | (s << 16));
                } else if (size == 2)
                        pan_pack_color_32(packed, out.ui[0] | (out.ui[0] << 16));
                else if (size == 3 || size == 4)
                        pan_pack_color_32(packed, out.ui[0]);
                else if (size == 6)
                        pan_pack_color_64(packed, out.ui[0], out.ui[1] | (out.ui[1] << 16)); /* RGB16F -- RGBB */
                else if (size == 8)
                        pan_pack_color_64(packed, out.ui[0], out.ui[1]);
                else if (size == 16)
                        memcpy(packed, out.ui, 16);
                else
                        unreachable("Unknown generic format size packing clear colour");
        }
}

void
panfrost_batch_clear(struct panfrost_batch *batch,
                     unsigned buffers,
                     const union pipe_color_union *color,
                     double depth, unsigned stencil)
{
        struct panfrost_context *ctx = batch->ctx;

        if (buffers & PIPE_CLEAR_COLOR) {
                for (unsigned i = 0; i < PIPE_MAX_COLOR_BUFS; ++i) {
                        if (!(buffers & (PIPE_CLEAR_COLOR0 << i)))
                                continue;

                        enum pipe_format format = ctx->pipe_framebuffer.cbufs[i]->format;
                        pan_pack_color(batch->clear_color[i], color, format);
                }
        }

        if (buffers & PIPE_CLEAR_DEPTH) {
                batch->clear_depth = depth;
        }

        if (buffers & PIPE_CLEAR_STENCIL) {
                batch->clear_stencil = stencil;
        }

        batch->clear |= buffers;

        /* Clearing affects the entire framebuffer (by definition -- this is
         * the Gallium clear callback, which clears the whole framebuffer. If
         * the scissor test were enabled from the GL side, the gallium frontend
         * would emit a quad instead and we wouldn't go down this code path) */

        panfrost_batch_union_scissor(batch, 0, 0,
                                     ctx->pipe_framebuffer.width,
                                     ctx->pipe_framebuffer.height);
}

static bool
panfrost_batch_compare(const void *a, const void *b)
{
        return util_framebuffer_state_equal(a, b);
}

static uint32_t
panfrost_batch_hash(const void *key)
{
        return _mesa_hash_data(key, sizeof(struct pipe_framebuffer_state));
}

/* Given a new bounding rectangle (scissor), let the job cover the union of the
 * new and old bounding rectangles */

void
panfrost_batch_union_scissor(struct panfrost_batch *batch,
                             unsigned minx, unsigned miny,
                             unsigned maxx, unsigned maxy)
{
        batch->minx = MIN2(batch->minx, minx);
        batch->miny = MIN2(batch->miny, miny);
        batch->maxx = MAX2(batch->maxx, maxx);
        batch->maxy = MAX2(batch->maxy, maxy);
}

void
panfrost_batch_intersection_scissor(struct panfrost_batch *batch,
                                  unsigned minx, unsigned miny,
                                  unsigned maxx, unsigned maxy)
{
        batch->minx = MAX2(batch->minx, minx);
        batch->miny = MAX2(batch->miny, miny);
        batch->maxx = MIN2(batch->maxx, maxx);
        batch->maxy = MIN2(batch->maxy, maxy);
}

/* Are we currently rendering to the dev (rather than an FBO)? */

bool
panfrost_batch_is_scanout(struct panfrost_batch *batch)
{
        /* If there is no color buffer, it's an FBO */
        if (batch->key.nr_cbufs != 1)
                return false;

        /* If we're too early that no framebuffer was sent, it's scanout */
        if (!batch->key.cbufs[0])
                return true;

        return batch->key.cbufs[0]->texture->bind & PIPE_BIND_DISPLAY_TARGET ||
               batch->key.cbufs[0]->texture->bind & PIPE_BIND_SCANOUT ||
               batch->key.cbufs[0]->texture->bind & PIPE_BIND_SHARED;
}

void
panfrost_batch_init(struct panfrost_context *ctx)
{
        ctx->batches = _mesa_hash_table_create(ctx,
                                               panfrost_batch_hash,
                                               panfrost_batch_compare);
        ctx->accessed_bos = _mesa_hash_table_create(ctx, _mesa_hash_pointer,
                                                    _mesa_key_pointer_equal);
}
