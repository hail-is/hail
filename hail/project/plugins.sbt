resolvers += "bintray-spark-packages" at "https://dl.bintray.com/spark-packages/maven/"
resolvers += Resolver.bintrayIvyRepo("s22s", "sbt-plugins")

addSbtPlugin("org.spark-packages" % "sbt-spark-package" % "0.2.7-astraea.1")
// addSbtPlugin("org.spark-packages" % "sbt-spark-package" % "0.2.6")
addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.14.6")
