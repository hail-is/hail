package is.hail.utils


trait PriorityQueue[T, R] {
  /**
    * Insert item {@code t} with priority {@code r}.
    *
    **/
  def insert(t: T, r: R): Unit

  /**
    * Returns the item with highest priority in this queue.
    *
    **/
  def max(): T

  /**
    * Returns the item with highest priority in this queue and removes it from
    * the queue.
    *
    **/
  def extractMax(): T

  /**
    * Set the priority of element {@code t} to {@code r}
    *
    **/
  def setPriority(t: T, r: R): Unit
}
