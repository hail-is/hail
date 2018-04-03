package is.hail.sparkextras

trait ResettableContext extends AutoCloseable {
  def reset(): Unit
}
