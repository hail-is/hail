package is.hail.backend

import is.hail.backend.service.WireProtocol
import is.hail.utils.{handleForPython, using, HailWorkerException}

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, DataInputStream, DataOutputStream}

import org.scalacheck.Prop.forAll

class WorkerSuite extends munit.ScalaCheckSuite {

  property("WriteReadSuccess") = forAll { (partitionId: Int, payload: Array[Byte]) =>
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

    readPartition == partitionId && java.util.Arrays.equals(result, payload)
  }

  property("WriteReadFailure") = forAll { (partitionId: Int, payload: Throwable) =>
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
    exception == HailWorkerException(
      partitionId = partitionId,
      shortMessage = short,
      expandedMessage = expanded,
      errorId = errorId,
    )
  }

}
