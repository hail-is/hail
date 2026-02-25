package is.hail.backend

import is.hail.asm4s.HailClassLoader

package object spark {
  // FIXME: how do I ensure this is only created in Spark workers?
  lazy val unsafeHailClassLoaderForSparkWorkers: HailClassLoader =
    new HailClassLoader(getClass.getClassLoader)
}
