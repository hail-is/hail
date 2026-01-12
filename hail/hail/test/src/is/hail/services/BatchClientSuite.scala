package is.hail.services

import is.hail.{services, Revision}
import is.hail.backend.service.Main
import is.hail.collection.FastSeq
import is.hail.services.JobGroupStates.Failure
import is.hail.services.oauth2.CloudCredentials
import is.hail.utils._

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future

import java.lang.reflect.Method
import java.nio.file.Path

import org.scalatest.Inspectors.forAll
import org.scalatest.enablers.InspectorAsserting.assertingNatureOfAssertion
import org.scalatest.matchers.should.Matchers.convertToAnyShouldWrapper
import org.scalatestplus.testng.TestNGSuite
import org.testng.annotations.{AfterClass, BeforeClass, BeforeMethod, Test}

class BatchClientSuite extends TestNGSuite {

  private[this] var client: BatchClient = _
  private[this] var batchId: Int = _
  private[this] var parentJobGroupId: Int = _

  @BeforeClass
  def createClientAndBatch(): Unit = {
    client =
      BatchClient(
        DeployConfig.default,
        CloudCredentials(Some(Path.of("/test-gsa-key/key.json"))),
      )

    batchId =
      client.newBatch(
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
    )._1
  }

  @AfterClass
  def closeClient(): Unit =
    client.close()

  @Test
  def testCancelAfterNFailures(): Unit = {
    val (jobGroupId, _) = client.newJobGroup(
      req = JobGroupRequest(
        batch_id = batchId,
        absolute_parent_id = parentJobGroupId,
        cancel_after_n_failures = Some(1),
        token = tokenUrlSafe,
        jobs = FastSeq(
          JobRequest(
            always_run = false,
            process = BashJob(
              image = "ubuntu:24.04",
              command = Array("/bin/bash", "-c", "sleep 5m"),
            ),
            resources = Some(JobResources(preemptible = true)),
          ),
          JobRequest(
            always_run = false,
            process = BashJob(
              image = "ubuntu:24.04",
              command = Array("/bin/bash", "-c", "exit 1"),
            ),
          ),
        ),
      )
    )
    val result = client.waitForJobGroup(batchId, jobGroupId)
    assert(result.state == Failure)
    assert(result.n_cancelled == 1)
  }

  @Test
  def testGetJobGroupJobsByState(): Unit = {
    val (jobGroupId, _) = client.newJobGroup(
      req = JobGroupRequest(
        batch_id = batchId,
        absolute_parent_id = parentJobGroupId,
        token = tokenUrlSafe,
        jobs = FastSeq(
          JobRequest(
            always_run = false,
            process = BashJob(
              image = "ubuntu:24.04",
              command = Array("/bin/bash", "-c", "exit 0"),
            ),
          ),
          JobRequest(
            always_run = false,
            process = BashJob(
              image = "ubuntu:24.04",
              command = Array("/bin/bash", "-c", "exit 1"),
            ),
          ),
        ),
      )
    )
    client.waitForJobGroup(batchId, jobGroupId): Unit
    forAll(Array(JobStates.Failed, JobStates.Success)) { state =>
      forAll(client.getJobGroupJobs(batchId, jobGroupId, Some(state))) { jobs =>
        assert(jobs.length == 1)
        assert(jobs(0).state == state)
      }
    }
  }

  @Test
  def testNewJobGroup(): Unit =
    // The query driver submits a job group per stage with one job per partition
    forAll(1 to 2) { i =>
      val (jobGroupId, _) = client.newJobGroup(
        req = JobGroupRequest(
          batch_id = batchId,
          absolute_parent_id = parentJobGroupId,
          token = tokenUrlSafe,
          attributes = Map("name" -> s"JobGroup$i"),
          jobs = (1 to i).map { k =>
            JobRequest(
              always_run = false,
              process = BashJob(
                image = "ubuntu:24.04",
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
    val (jobGroupId, _) = client.newJobGroup(
      req = JobGroupRequest(
        batch_id = batchId,
        absolute_parent_id = parentJobGroupId,
        token = tokenUrlSafe,
        attributes = Map("name" -> "TableStage"),
        jobs = FastSeq(
          JobRequest(
            always_run = false,
            process = JvmJob(
              command = Array(Main.TEST),
              spec = GitRevision(Revision),
              profile = false,
            ),
          )
        ),
      )
    )

    val result = client.getJobGroup(batchId, jobGroupId)
    assert(result.n_jobs == 1)
  }

  @Test // #15118
  def testWaitForCancelledJobGroup(): Unit = {
    val (jobGroupId, _) = client.newJobGroup(
      req = JobGroupRequest(
        batch_id = batchId,
        absolute_parent_id = parentJobGroupId,
        cancel_after_n_failures = Some(1),
        token = tokenUrlSafe,
        jobs = 0 until 10 map { _ =>
          JobRequest(
            always_run = false,
            process = BashJob(
              image = "ubuntu:24.04",
              command = Array("/bin/bash", "-c", "sleep 5s"),
            ),
            resources = Some(JobResources(preemptible = true)),
          )
        },
      )
    )

    Future {
      Thread.sleep(100)
      client.cancelJobGroup(batchId, jobGroupId)
    }: Unit

    val jg = client.waitForJobGroup(batchId, jobGroupId)
    jg.state shouldBe services.JobGroupStates.Cancelled
  }
}
