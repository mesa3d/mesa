#include "zink_query.h"

#include "zink_context.h"
#include "zink_resource.h"
#include "zink_screen.h"

#include "util/u_dump.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"

struct zink_query {
   enum pipe_query_type type;

   VkQueryPool query_pool;
   unsigned curr_query, num_queries;

   VkQueryType vkqtype;
   bool use_64bit;
   bool precise;

   struct list_head active_list;
};

static VkQueryType
convert_query_type(unsigned query_type, bool *use_64bit, bool *precise)
{
   *use_64bit = false;
   *precise = false;
   switch (query_type) {
   case PIPE_QUERY_OCCLUSION_COUNTER:
      *precise = true;
      *use_64bit = true;
   case PIPE_QUERY_OCCLUSION_PREDICATE:
   case PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE:
      return VK_QUERY_TYPE_OCCLUSION;
   case PIPE_QUERY_TIMESTAMP:
      *use_64bit = true;
      return VK_QUERY_TYPE_TIMESTAMP;
   case PIPE_QUERY_PIPELINE_STATISTICS:
      return VK_QUERY_TYPE_PIPELINE_STATISTICS;
   default:
      debug_printf("unknown query: %s\n",
                   util_str_query_type(query_type, true));
      unreachable("zink: unknown query type");
   }
}

static struct pipe_query *
zink_create_query(struct pipe_context *pctx,
                  unsigned query_type, unsigned index)
{
   struct zink_screen *screen = zink_screen(pctx->screen);
   struct zink_query *query = CALLOC_STRUCT(zink_query);
   VkQueryPoolCreateInfo pool_create = {};

   if (!query)
      return NULL;

   query->type = query_type;
   query->vkqtype = convert_query_type(query_type, &query->use_64bit, &query->precise);
   if (query->vkqtype == -1)
      return NULL;

   query->num_queries = query_type == PIPE_QUERY_TIMESTAMP ? 1 : 100;
   query->curr_query = 0;

   pool_create.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
   pool_create.queryType = query->vkqtype;
   pool_create.queryCount = query->num_queries;

   VkResult status = vkCreateQueryPool(screen->dev, &pool_create, NULL, &query->query_pool);
   if (status != VK_SUCCESS) {
      FREE(query);
      return NULL;
   }
   return (struct pipe_query *)query;
}

/* TODO: rework this to be less hammer-ish using deferred destroy */
static void
wait_query(struct pipe_context *pctx, struct zink_query *query)
{
   struct pipe_fence_handle *fence = NULL;

   pctx->flush(pctx, &fence, PIPE_FLUSH_HINT_FINISH);
   if (fence) {
      pctx->screen->fence_finish(pctx->screen, NULL, fence,
                                 PIPE_TIMEOUT_INFINITE);
      pctx->screen->fence_reference(pctx->screen, &fence, NULL);
   }
}

static void
zink_destroy_query(struct pipe_context *pctx,
                   struct pipe_query *q)
{
   struct zink_screen *screen = zink_screen(pctx->screen);
   struct zink_query *query = (struct zink_query *)q;

   if (!list_is_empty(&query->active_list)) {
      wait_query(pctx, query);
   }

   vkDestroyQueryPool(screen->dev, query->query_pool, NULL);
   FREE(query);
}

static void
begin_query(struct zink_context *ctx, struct zink_query *q)
{
   VkQueryControlFlags flags = 0;
   if (q->precise)
      flags |= VK_QUERY_CONTROL_PRECISE_BIT;

   struct zink_batch *batch = zink_curr_batch(ctx);
   vkCmdBeginQuery(batch->cmdbuf, q->query_pool, q->curr_query, flags);
}

static bool
zink_begin_query(struct pipe_context *pctx,
                 struct pipe_query *q)
{
   struct zink_context *ctx = zink_context(pctx);
   struct zink_query *query = (struct zink_query *)q;

   /* ignore begin_query for timestamps */
   if (query->type == PIPE_QUERY_TIMESTAMP)
      return true;

   /* TODO: resetting on begin isn't ideal, as it forces render-pass exit...
    * should instead reset on creation (if possible?)... Or perhaps maintain
    * the pool in the batch instead?
    */
   struct zink_batch *batch = zink_batch_no_rp(zink_context(pctx));
   vkCmdResetQueryPool(batch->cmdbuf, query->query_pool, 0, MIN2(query->curr_query + 1, query->num_queries));
   query->curr_query = 0;

   begin_query(ctx, query);
   list_addtail(&query->active_list, &ctx->active_queries);

   return true;
}

static void
end_query(struct zink_context *ctx, struct zink_query *q)
{
   struct zink_batch *batch = zink_curr_batch(ctx);
   assert(q->type != PIPE_QUERY_TIMESTAMP);
   vkCmdEndQuery(batch->cmdbuf, q->query_pool, q->curr_query);
   if (++q->curr_query == q->num_queries) {
      assert(0);
      /* need to reset pool! */
   }
}

static bool
zink_end_query(struct pipe_context *pctx,
               struct pipe_query *q)
{
   struct zink_context *ctx = zink_context(pctx);
   struct zink_query *query = (struct zink_query *)q;

   if (query->type == PIPE_QUERY_TIMESTAMP) {
      assert(query->curr_query == 0);
      struct zink_batch *batch = zink_curr_batch(ctx);
      vkCmdWriteTimestamp(batch->cmdbuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                          query->query_pool, 0);
   } else {
      end_query(ctx, query);
      list_delinit(&query->active_list);
   }

   return true;
}

static bool
zink_get_query_result(struct pipe_context *pctx,
                      struct pipe_query *q,
                      bool wait,
                      union pipe_query_result *result)
{
   struct zink_screen *screen = zink_screen(pctx->screen);
   struct zink_query *query = (struct zink_query *)q;
   VkQueryResultFlagBits flags = 0;

   if (wait) {
      wait_query(pctx, query);
      flags |= VK_QUERY_RESULT_WAIT_BIT;
   } else
      pctx->flush(pctx, NULL, 0);

   if (query->use_64bit)
      flags |= VK_QUERY_RESULT_64_BIT;

   // TODO: handle curr_query > 100
   // union pipe_query_result results[100];
   uint64_t results[100];
   memset(results, 0, sizeof(results));
   assert(query->curr_query <= ARRAY_SIZE(results));
   if (vkGetQueryPoolResults(screen->dev, query->query_pool,
                             0, query->curr_query,
                             sizeof(results),
                             results,
                             sizeof(uint64_t),
                             flags) != VK_SUCCESS)
      return false;

   util_query_clear_result(result, query->type);
   for (int i = 0; i < query->curr_query; ++i) {
      switch (query->type) {
      case PIPE_QUERY_OCCLUSION_PREDICATE:
      case PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE:
      case PIPE_QUERY_SO_OVERFLOW_PREDICATE:
      case PIPE_QUERY_SO_OVERFLOW_ANY_PREDICATE:
      case PIPE_QUERY_GPU_FINISHED:
         result->b |= results[i] != 0;
         break;

      case PIPE_QUERY_OCCLUSION_COUNTER:
         result->u64 += results[i];
         break;

      default:
         debug_printf("unhangled query type: %s\n",
                      util_str_query_type(query->type, true));
         unreachable("unexpected query type");
      }
   }

   return TRUE;
}

void
zink_suspend_queries(struct zink_context *ctx, struct zink_batch *batch)
{
   struct zink_query *query;
   LIST_FOR_EACH_ENTRY(query, &ctx->active_queries, active_list) {
      end_query(ctx, query);
   }
}

void
zink_resume_queries(struct zink_context *ctx, struct zink_batch *batch)
{
   struct zink_query *query;
   LIST_FOR_EACH_ENTRY(query, &ctx->active_queries, active_list) {
      vkCmdResetQueryPool(batch->cmdbuf, query->query_pool, query->curr_query, 1);
      begin_query(ctx, query);
   }
}

static void
zink_set_active_query_state(struct pipe_context *pctx, bool enable)
{
   struct zink_context *ctx = zink_context(pctx);
   ctx->queries_disabled = !enable;

   struct zink_batch *batch = zink_curr_batch(ctx);
   if (ctx->queries_disabled)
      zink_suspend_queries(ctx, batch);
   else
      zink_resume_queries(ctx, batch);
}

static void
zink_render_condition(struct pipe_context *pctx,
                      struct pipe_query *pquery,
                      bool condition,
                      enum pipe_render_cond_flag mode)
{
   struct zink_context *ctx = zink_context(pctx);
   struct zink_screen *screen = zink_screen(pctx->screen);
   struct zink_query *query = (struct zink_query *)pquery;
   struct zink_batch *batch = zink_curr_batch(ctx);
   VkQueryResultFlagBits flags = 0;

   if (query == NULL) {
      screen->vk_CmdEndConditionalRenderingEXT(batch->cmdbuf);
      return;
   }

   struct pipe_resource *pres;
   struct zink_resource *res;
   struct pipe_resource templ = {};
   templ.width0 = 8;
   templ.height0 = 1;
   templ.depth0 = 1;
   templ.format = PIPE_FORMAT_R8_UINT;
   templ.target = PIPE_BUFFER;

   /* need to create a vulkan buffer to copy the data into */
   pres = pctx->screen->resource_create(pctx->screen, &templ);
   if (!pres)
      return;

   res = (struct zink_resource *)pres;

   if (mode == PIPE_RENDER_COND_WAIT || mode == PIPE_RENDER_COND_BY_REGION_WAIT)
      flags |= VK_QUERY_RESULT_WAIT_BIT;

   if (query->use_64bit)
      flags |= VK_QUERY_RESULT_64_BIT;
   vkCmdCopyQueryPoolResults(batch->cmdbuf, query->query_pool, 0, 1,
                             res->buffer, 0, 0, flags);

   VkConditionalRenderingFlagsEXT begin_flags = 0;
   if (condition)
      begin_flags = VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT;
   VkConditionalRenderingBeginInfoEXT begin_info = {};
   begin_info.sType = VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT;
   begin_info.buffer = res->buffer;
   begin_info.flags = begin_flags;
   screen->vk_CmdBeginConditionalRenderingEXT(batch->cmdbuf, &begin_info);

   zink_batch_reference_resoure(batch, res);

   pipe_resource_reference(&pres, NULL);
}

void
zink_context_query_init(struct pipe_context *pctx)
{
   struct zink_context *ctx = zink_context(pctx);
   list_inithead(&ctx->active_queries);

   pctx->create_query = zink_create_query;
   pctx->destroy_query = zink_destroy_query;
   pctx->begin_query = zink_begin_query;
   pctx->end_query = zink_end_query;
   pctx->get_query_result = zink_get_query_result;
   pctx->set_active_query_state = zink_set_active_query_state;
   pctx->render_condition = zink_render_condition;
}
