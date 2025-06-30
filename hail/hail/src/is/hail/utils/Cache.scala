package is.hail.utils

import is.hail.annotations.{Region, RegionMemory}
import is.hail.macros.void

import scala.collection.mutable
import scala.jdk.CollectionConverters.asScalaIteratorConverter

import java.io.Closeable
import java.util
import java.util.Map.Entry

class Cache[K, V](capacity: Int) extends mutable.AbstractMap[K, V] {
  private[this] val m = new util.LinkedHashMap[K, V](capacity, 0.75f, true) {
    override def removeEldestEntry(eldest: Entry[K, V]): Boolean = size() > capacity
  }

  override def +=(kv: (K, V)): Cache.this.type =
    synchronized { m.put(kv._1, kv._2); this }

  override def -=(key: K): Cache.this.type =
    synchronized { m.remove(key); this }

  override def get(key: K): Option[V] =
    synchronized(Option(m.get(key)))

  override def iterator: Iterator[(K, V)] =
    for { e <- m.entrySet().iterator().asScala } yield (e.getKey, e.getValue)

  override def clear(): Unit =
    m.clear()
}

class LongToRegionValueCache(capacity: Int) extends Closeable {
  private[this] val m = new util.LinkedHashMap[Long, (RegionMemory, Long)](capacity, 0.75f, true) {
    override def removeEldestEntry(eldest: Entry[Long, (RegionMemory, Long)]): Boolean = {
      val b = (size() > capacity)
      if (b) {
        val (rm, _) = eldest.getValue
        rm.release()
      }
      b
    }
  }

  // the cache takes ownership of the region passed in
  def put(key: Long, region: Region, addr: Long): Unit = {
    if (addr == 0L)
      throw new RuntimeException("tried to cache null pointer")
    val rm = region.getMemory()
    void(m.put(key, (rm, addr)))
  }

  // returns -1 if not in cache
  def get(key: Long): Long = {
    val v = m.get(key)
    if (v == null)
      0L
    else
      v._2
  }

  def free(): Unit = {
    m.forEach((_, v) => v._1.release())
    m.clear()
  }

  def close(): Unit = free()
}
