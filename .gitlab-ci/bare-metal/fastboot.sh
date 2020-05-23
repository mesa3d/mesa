#!/bin/bash

BM=$CI_PROJECT_DIR/.gitlab-ci/bare-metal

if [ -z "$BM_SERIAL" ]; then
  echo "Must set BM_SERIAL in your gitlab-runner config.toml [[runners]] environment"
  echo "This is the serial device to talk to for waiting for fastboot to be ready and logging from the kernel."
  exit 1
fi

if [ -z "$BM_POWERUP" ]; then
  echo "Must set BM_POWERUP in your gitlab-runner config.toml [[runners]] environment"
  echo "This is a shell script that should reset the device and begin its boot sequence"
  echo "such that it pauses at fastboot."
  exit 1
fi

if [ -z "$BM_POWERDOWN" ]; then
  echo "Must set BM_POWERDOWN in your gitlab-runner config.toml [[runners]] environment"
  echo "This is a shell script that should power off the device."
  exit 1
fi

if [ -z "$BM_FASTBOOT_SERIAL" ]; then
  echo "Must set BM_FASTBOOT_SERIAL in your gitlab-runner config.toml [[runners]] environment"
  echo "This must be the a stable-across-resets fastboot serial number."
  exit 1
fi

if [ -z "$BM_KERNEL" ]; then
  echo "Must set BM_KERNEL to your board's kernel vmlinuz or Image.gz in the job's variables:"
  exit 1
fi

if [ -z "$BM_DTB" ]; then
  echo "Must set BM_DTB to your board's DTB file in the job's variables:"
  exit 1
fi

if [ -z "$BM_ROOTFS" ]; then
  echo "Must set BM_ROOTFS to your board's rootfs directory in the job's variables:"
  exit 1
fi

set -ex

# Copy the rootfs to a temporary for our setup, as I believe changes to the
# container can end up impacting future runs.
cp -Rp $BM_ROOTFS rootfs

# Set up the init script that brings up the system.
cp $BM/init.sh rootfs/init

set +x
# Pass through relevant env vars from the gitlab job to the baremetal init script
touch rootfs/set-job-env-vars.sh
chmod +x rootfs/set-job-env-vars.sh
for var in \
    CI_COMMIT_BRANCH \
    CI_COMMIT_TITLE \
    CI_JOB_ID \
    CI_JOB_URL \
    CI_MERGE_REQUEST_SOURCE_BRANCH_NAME \
    CI_MERGE_REQUEST_TITLE \
    CI_NODE_INDEX \
    CI_NODE_TOTAL \
    CI_PIPELINE_ID \
    CI_RUNNER_DESCRIPTION \
    DEQP_CASELIST_FILTER \
    DEQP_EXPECTED_RENDERER \
    DEQP_PARALLEL \
    DEQP_RUN_SUFFIX \
    DEQP_VER \
    FD_MESA_DEBUG \
    FLAKES_CHANNEL \
    IR3_SHADER_DEBUG \
    NIR_VALIDATE \
    ; do
  val=`echo ${!var} | sed 's|"||g'`
  echo "export $var=\"${val}\"" >> rootfs/set-job-env-vars.sh
done
echo "Variables passed through:"
cat rootfs/set-job-env-vars.sh
set -x

# Add the Mesa drivers we built, and make a consistent symlink to them.
mkdir -p rootfs/$CI_PROJECT_DIR
tar -C rootfs/$CI_PROJECT_DIR/ -xf $CI_PROJECT_DIR/artifacts/install.tar
ln -sf $CI_PROJECT_DIR/install rootfs/install

# Copy the deqp runner script and metadata.
cp .gitlab-ci/deqp-runner.sh rootfs/deqp/.
cp .gitlab-ci/$DEQP_SKIPS rootfs/$CI_PROJECT_DIR/install/deqp-skips.txt
if [ -n "$DEQP_EXPECTED_FAILS" ]; then
  cp .gitlab-ci/$DEQP_EXPECTED_FAILS rootfs/$CI_PROJECT_DIR/install/deqp-expected-fails.txt
fi

# Finally, pack it up into a cpio rootfs.
pushd rootfs
  find -H | cpio -H newc -o | xz --check=crc32 -T4 - > $CI_PROJECT_DIR/rootfs.cpio.gz
popd

cat $BM_KERNEL $BM_DTB > Image.gz-dtb

abootimg \
  --create artifacts/fastboot.img \
  -k Image.gz-dtb \
  -r rootfs.cpio.gz \
  -c cmdline="$BM_CMDLINE"
rm Image.gz-dtb

# Start watching serial, and power up the device.
$BM/serial-buffer.py $BM_SERIAL | tee artifacts/serial-output.txt &
while [ ! -e artifacts/serial-output.txt ]; do
  sleep 1
done
PATH=$BM:$PATH $BM_POWERUP

# Once fastboot is ready, boot our image.
$BM/expect-output.sh artifacts/serial-output.txt "fastboot: processing commands"
fastboot boot -s $BM_FASTBOOT_SERIAL artifacts/fastboot.img

# Wait for the device to complete the deqp run
$BM/expect-output.sh artifacts/serial-output.txt "DEQP RESULT"

# power down the device
PATH=$BM:$PATH $BM_POWERDOWN

set +e
if grep -q "DEQP RESULT: pass" artifacts/serial-output.txt; then
   exit 0
else
   exit 1
fi

