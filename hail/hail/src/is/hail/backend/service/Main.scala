package is.hail.backend.service

import is.hail.backend.driver.BatchQueryDriver

import com.fasterxml.jackson.core.StreamReadConstraints

object Main {
  val WORKER = "worker"
  val DRIVER = "driver"
  val TEST = "test"

  /* This constraint should be overridden as early as possible. See:
   * - https://github.com/hail-is/hail/issues/14580
   * - https://github.com/hail-is/hail/issues/14749
   * - The note on this setting in is.hail.backend.Backend companion objects */
  StreamReadConstraints.overrideDefaultStreamReadConstraints(
    StreamReadConstraints.builder().maxStringLength(Integer.MAX_VALUE).build()
  )

  is.hail.linalg.registerImplOpMulMatrix_DMD_DVD_eq_DVD

  def main(argv: Array[String]): Unit =
    argv(3) match {
      case WORKER => Worker.main(argv)
      case DRIVER => BatchQueryDriver.main(argv)
      // Batch's "JvmJob" is a special kind of job that can only call `Main.main`.
      // TEST is used for integration testing the `BatchClient` to verify that we
      // can create JvmJobs without having to mock the payload to a `Worker` job.
      case TEST => ()
      case kind => throw new RuntimeException(s"unknown kind: $kind")
    }
}
