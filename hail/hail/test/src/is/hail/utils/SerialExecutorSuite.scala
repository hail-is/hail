package is.hail.utils

import java.util.concurrent.CountDownLatch

import org.junit.jupiter.api.Test
import org.scalatest.matchers.should.Matchers._

class SerialExecutorSuite {

  private[this] def withExecutor[A](test: SerialExecutor => A): A = {
    val executor = new SerialExecutor(getClass.getName)
    try test(executor)
    finally executor.close()
  }

  // run `op` on the executor from another thread, returning after `op` has
  // started; join the result to observe what the submitter saw
  private[this] def submitAndAwaitStart(executor: SerialExecutor)(op: => Unit): Thread = {
    val started = new CountDownLatch(1)
    val submitter = new Thread(() =>
      try
        executor.run {
          started.countDown()
          op
        }
      catch { case _: HailException => }
    )
    submitter.start()
    started.await()
    submitter
  }

  @Test def testRejectsConcurrentOperations(): Unit =
    withExecutor { executor =>
      val release = new CountDownLatch(1)
      val busy = submitAndAwaitStart(executor)(release.await())
      try
        the[HailException] thrownBy executor.run(()) should have message
          "another operation is in progress; wait for it to finish or cancel it"
      finally {
        release.countDown()
        busy.join()
      }
    }

  @Test def testCancelInterruptsTheRunningOperation(): Unit =
    withExecutor { executor =>
      @volatile var failure: Throwable = null
      val started = new CountDownLatch(1)
      val submitter = new Thread(() =>
        try
          executor.run {
            started.countDown()
            Thread.sleep(Long.MaxValue)
          }
        catch { case t: Throwable => failure = t }
      )
      submitter.start()
      started.await()
      executor.cancel()
      submitter.join()
      failure should have message "operation was cancelled"
    }

  @Test def testCancelIsANoOpWhenIdle(): Unit =
    withExecutor { executor =>
      executor.cancel()
      executor.run(42) shouldBe 42
    }

  @Test def testAcceptsNewOperationsAfterCancellation(): Unit =
    withExecutor { executor =>
      val cancelled = submitAndAwaitStart(executor)(Thread.sleep(Long.MaxValue))
      executor.cancel()
      cancelled.join()
      executor.run(42) shouldBe 42
    }
}
