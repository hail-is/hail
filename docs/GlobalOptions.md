# Global Options

Hail global command line options:
 - `-a | --log-append` -- Append to log file instead of overwritting.
 - `-h | --help` -- Print usage and exit.
 - `-l <log file> | --log-file <log file>` -- Log file.  Default: hail.log.
 - `--master <master>` -- Set Spark master.  Default: system default or local[*].
 - `--parquet-compression <codec>` -- Parquet compression codec.  Default: uncompressed.
 - `-q | --quiet` -- Don't write log file.
 - `--stacktrace` -- Print stacktrace on unhandled exception.  For developers.
 - `-t <tmpdir> | --tmpdir <tmpdir>` -- Temporary directory.  Default: /tmp.
