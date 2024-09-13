package is.hail.services

import is.hail.utils._

import java.nio.file.Path

import org.scalatestplus.testng.TestNGSuite
import org.testng.annotations.Test

class BatchClientSuite extends TestNGSuite {
  @Test def testBasic(): Unit =
    using(BatchClient(DeployConfig.get(), Path.of("/test-gsa-key/key.json"))) { client =>
      val jobGroup = client.run(
        BatchRequest(
          billing_project = "test",
          n_jobs = 1,
          token = tokenUrlSafe,
        ),
        JobGroupRequest(
          job_group_id = 0,
          absolute_parent_id = 0,
        ),
        FastSeq(
          JobRequest(
            job_id = 0,
            always_run = false,
            in_update_job_group_id = 0,
            in_update_parent_ids = Array(),
            process = BashJob(
              image = "ubuntu:22.04",
              command = Array("/bin/bash", "-c", "'hello, world!"),
            ),
          )
        ),
      )

      assert(jobGroup.state == JobGroupStates.Success)
    }
}
