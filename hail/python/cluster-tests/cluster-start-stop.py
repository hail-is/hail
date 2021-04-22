import hail as hl

# turn it on
hl.init()

# turn it off
hl.stop()

# turn it back on (Google Dataproc with Spark 3 has issues immediately restarting context after stopping, some kind of race condition in yarn 3)
# hl.init()
