package is.hail.services

import is.hail.backend.service.Main
import is.hail.utils._

import scala.sys.process._

import java.lang.reflect.Method
import java.nio.file.Path

import org.scalatestplus.testng.TestNGSuite
import org.testng.annotations.{AfterClass, BeforeClass, BeforeMethod, Test}

class BatchClientSuite extends TestNGSuite {

  private[this] var client: BatchClient = _
  private[this] var batchId: Int = _
  private[this] var parentJobGroupId: Int = _

  @BeforeClass
  def createClientAndBatch(): Unit = {
    client = BatchClient(DeployConfig.get(), Path.of("/tmp/test-gsa-key/key.json"))
    batchId = client.newBatch(
      BatchRequest(
        billing_project = "test",
        n_jobs = 0,
        token = tokenUrlSafe,
        attributes = Map("name" -> s"${getClass.getName}"),
      )
    )
  }

  @BeforeMethod
  def createEmptyParentJobGroup(m: Method): Unit = {
    parentJobGroupId = client.newJobGroup(
      req = JobGroupRequest(
        batch_id = batchId,
        absolute_parent_id = 0,
        token = tokenUrlSafe,
        attributes = Map("name" -> m.getName),
        jobs = FastSeq(),
      )
    )
  }

  @AfterClass
  def closeClient(): Unit =
    client.close()

  @Test
  def testNewJobGroup(): Unit =
    // The query driver submits a job group per stage with one job per partition
    for (i <- 1 to 2) {
      val jobGroupId = client.newJobGroup(
        req = JobGroupRequest(
          batch_id = batchId,
          absolute_parent_id = parentJobGroupId,
          token = tokenUrlSafe,
          attributes = Map("name" -> s"JobGroup$i"),
          jobs = (1 to i).map { k =>
            JobRequest(
              always_run = false,
              process = BashJob(
                image = "ubuntu:22.04",
                command = Array("/bin/bash", "-c", s"echo 'job $k'"),
              ),
            )
          },
        )
      )

      val result = client.getJobGroup(batchId, jobGroupId)
      assert(result.n_jobs == i)
    }

  @Test
  def testJvmJob(): Unit = {
    val jobGroupId = client.newJobGroup(
      req = JobGroupRequest(
        batch_id = batchId,
        absolute_parent_id = parentJobGroupId,
        token = tokenUrlSafe,
        attributes = Map("name" -> "TableStage"),
        jobs = FastSeq(
          JobRequest(
            always_run = false,
            process = JvmJob(
              command = Array(Main.WORKER, "", "", ""),
              spec = GitRevision("git rev-parse main".!!.strip()),
              profile = false,
            ),
          )
        ),
      )
    )

    val result = client.getJobGroup(batchId, jobGroupId)
    assert(result.n_jobs == 1)
  }
}
