package is.hail.scheduler

import org.scalatest.testng.TestNGSuite
import org.testng.SkipException
import org.testng.annotations.Test
import is.hail.TestUtils._

class SchedulerTestException(message: String) extends Exception(message)

class SchedulerSuite extends TestNGSuite {
  private val schedulerHost = sys.env.get("HAIL_TEST_SCHEDULER_HOST") match {
    case Some(host) => host
    case None => "localhost"
  }

  @Test def testSimpleJob(): Unit = {
    if (sys.env.contains("HAIL_TEST_SKIP_SCHEDULER")) {
      throw new SkipException("Skipping tests of the scheduler")
    }

    val a = Array(-5, 6, 11)
    val da: DArray[Int] = new DArray[Int] {
      type Context = Int
      val contexts: Array[Int] = a
      val body: Int => Int = c => c
    }

    val conn = new SchedulerAppClient(schedulerHost)

    var n = 0
    val r = new Array[Int](3)
    conn.submit(da,
      (i: Int, x: Int) => {
        n += 1
        r(i) = x
      })
    assert(n == 3)
    assert(r sameElements a)

    conn.close()
  }

  @Test def testException(): Unit = {
    if (sys.env.contains("HAIL_TEST_SKIP_SCHEDULER")) {
      throw new SkipException("Skipping tests of the scheduler")
    }

    val da: DArray[Int] = new DArray[Int] {
      type Context = Int
      val contexts: Array[Int] = Array(1, 2, 3)
      val body: Int => Int = c => {
        if (c == 2)
          throw new SchedulerTestException("test message")
        else
          c
      }
    }

    val conn = new SchedulerAppClient(schedulerHost)

    interceptException[SchedulerTestException]("test message") {
      conn.submit(da,
        (i: Int, x: Int) => {
          assert(i == x - 1)
          assert(i == 0 || i == 2)
        })
    }

    conn.close()
  }
}
