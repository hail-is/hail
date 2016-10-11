## <a class="jumptarget" name="errormessages"></a> Error Messages

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
