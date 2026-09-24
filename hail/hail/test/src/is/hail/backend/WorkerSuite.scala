package is.hail.backend

import is.hail.TestUtils._
import is.hail.backend.service.WireProtocol
import is.hail.utils.{handleForPython, using, HailWorkerException}

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, DataInputStream, DataOutputStream}

import org.junit.jupiter.api.Test
import org.scalacheck.Prop.forAll
import org.scalatest.matchers.should.Matchers.convertToAnyShouldWrapper

class WorkerSuite {

  @Test def testWriteReadSuccess(): Unit =
    check(forAll { (partitionId: Int, payload: Array[Byte]) =>
      val buffer =
        using(new ByteArrayOutputStream()) { bs =>
          using(new DataOutputStream(bs)) { os =>
            WireProtocol.write(os, partitionId, Right(payload))
            bs.toByteArray
          }
        }

      val (result, readPartition) =
        using(new ByteArrayInputStream(buffer)) { bs =>
          using(new DataInputStream(bs))(is => WireProtocol.read(is).getOrElse(null))
        }

      readPartition shouldBe partitionId
      result shouldBe payload
    })

  @Test def testWriteReadFailure(): Unit =
    check(forAll { (partitionId: Int, payload: Throwable) =>
      val buffer =
        using(new ByteArrayOutputStream()) { bs =>
          using(new DataOutputStream(bs)) { os =>
            WireProtocol.write(os, partitionId, Left(payload))
            bs.toByteArray
          }
        }

      val exception: HailWorkerException =
        using(new ByteArrayInputStream(buffer)) { bs =>
          using(new DataInputStream(bs))(is => WireProtocol.read(is).left.getOrElse(null))
        }

      val (short, expanded, errorId) = handleForPython(payload)
      exception shouldBe HailWorkerException(
        partitionId = partitionId,
        shortMessage = short,
        expandedMessage = expanded,
        errorId = errorId,
      )
    })

}
