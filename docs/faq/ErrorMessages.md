## <a class="jumptarget" name="errormessages"></a> Error Messages

### Spark

#### How do I fix `Exception in thread "main" java.net.BindException: Can't assign requested address ...`?

The full error message should look similar to this:

```
Exception in thread "main" java.net.BindException: Can't assign requested address: Service 'sparkDriver' failed after 16 retries! Consider explicitly setting the appropriate port for the service 'sparkDriver' (for example spark.ui.port for SparkUI) to an available port or increasing spark.port.maxRetries.
```

This error is often caused by running spark on a machine connected to a VPN or a personal wi-fi hotspot (i.e. tethering to a phone). First, try fixing it by editing your hosts file as suggested in [this StackOverflow post](http://stackoverflow.com/a/35852781/6823256). If that fails, try running `hail` while disconnected from any VPN or personal wi-fi hotspot.

#### When running Hail locally, how do I set the Spark local directory (scratch space) to something other than the default `/tmp`?

Set this environment variable with `export SPARK_LOCAL_DIRS=/path/to/myTmp` before running Hail, which should have write permission to `myTmp`. This resolves I/O errors that arise when `/tmp` has too little space or is cleaned by the system while Hail is running.

### IntelliJ

#### How do I fix this error message in IntelliJ: `Error:Cause: org.gradle.api.tasks.scala.ScalaCompileOptions.getForce()Ljava/lang/String;`?

Enter these commands in the Hail directory:

```
rm -r .idea/
rm hail.iml
```

And then reimport the Hail project from the `build.gradle` file.

#### How do I fix an error message about Java Target 1.8 in IntelliJ?

1. Click on Preferences > Build, Execution, Deployment > Scala Compiler
2. Click on hail
3. Change additional compiler options from `-target:jvm-1.8` to `-target:javm-1.7`
