resolvers += "bintray-spark-packages" at "https://dl.bintray.com/spark-packages/maven/"
resolvers += Resolver.bintrayIvyRepo("s22s", "sbt-plugins")

addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.14.6")
