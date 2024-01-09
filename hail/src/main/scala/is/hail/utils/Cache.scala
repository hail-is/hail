package is.hail.utils

import is.hail.annotations.{Region, RegionMemory}

import java.io.Closeable
import java.util
import java.util.Map.Entry

class Cache[K, V](capacity: Int) {
  private[this] val m = new util.LinkedHashMap[K, V](capacity, 0.75f, true) {
    override def removeEldestEntry(eldest: Entry[K, V]): Boolean = size() > capacity
  }

  def get(k: K): Option[V] = synchronized(Option(m.get(k)))

  def +=(p: (K, V)): Unit = synchronized(m.put(p._1, p._2))

  def size: Int = synchronized(m.size())
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
    m.put(key, (rm, addr))
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
    m.forEach((k, v) => v._1.release())
    m.clear()
  }

  def close(): Unit = free()
}
