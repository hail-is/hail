package is.hail.services.batch_client

import is.hail.utils._
import com.jsoniter
import org.apache.commons.io.IOUtils
import org.apache.http.HttpEntity
import org.apache.http.client.methods.{CloseableHttpResponse, HttpDelete, HttpGet, HttpPatch, HttpPost, HttpUriRequest}
import org.apache.http.impl.client.{CloseableHttpClient, HttpClients}

class NoBodyException(message: String, cause: Throwable) extends Exception(message, cause) {
  def this() = this(null, null)

  def this(message: String) = this(message, null)
}

class BatchClient extends AutoCloseable {
   val httpClient: CloseableHttpClient = HttpClients.createDefault();

  private[this] def request(req: HttpUriRequest): jsoniter.any.Any = {
    using(httpClient.execute(req)) { resp =>
      val  entity: HttpEntity = resp.getEntity
      if (entity == null)
        throw new NoBodyException()
      using(entity.getContent) { content =>
        val s = IOUtils.toByteArray(content)
        jsoniter.JsonIterator.deserialize(s)
      }
    }
  }

  def get(path: String): jsoniter.any.Any =
    request(new HttpGet(s"https://batch.hail.is$path"))

  def post(path: String): jsoniter.any.Any =
    request(new HttpPost(s"https://batch.hail.is$path"))

  def patch(path: String): jsoniter.any.Any =
    request(new HttpPatch(s"https://batch.hail.is$path"))

  def delete(path: String): jsoniter.any.Any =
    request(new HttpDelete(s"https://batch.hail.is$path"))
  
  def close() {
    httpClient.close()
  }
}
