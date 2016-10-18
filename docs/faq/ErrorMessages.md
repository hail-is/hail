## <a class="jumptarget" name="errormessages"></a> Error Messages

### Spark

#### How do I fix `Exception in thread "main" java.net.BindException: Can't assign requested address ...`?

The full error message should look similar to this:

```
Exception in thread "main" java.net.BindException: Can't assign requested address: Service 'sparkDriver' failed after 16 retries! Consider explicitly setting the appropriate port for the service 'sparkDriver' (for example spark.ui.port for SparkUI) to an available port or increasing spark.port.maxRetries.
```

This error is often caused by running spark on a machine connected to a VPN or a personal wi-fi hotspot (i.e. tethering to a phone). Try running `hail` while disconnected from any VPN or personal wi-fi hotspot.

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
