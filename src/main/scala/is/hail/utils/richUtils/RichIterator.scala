package is.hail.utils.richUtils

import java.io.PrintWriter

import is.hail.utils.{SharedIterable, SharedIterator}

import scala.collection.JavaConverters._
import scala.io.Source

class RichIterator[T](val it: Iterator[T]) extends AnyVal {
  def toSharedIterator: SharedIterator[T] = new SharedIterator[T] {
    def hasNext: Boolean = it.hasNext

    def next(): T = it.next()
  }

  def flatMap[U](f: T => SharedIterable[U]): SharedIterable[U] = new SharedIterable[U] {
    def iterator: SharedIterator[U] = new SharedIterator[U] {
      self =>

      private var cur: SharedIterator[U] = SharedIterator.empty

      private def nextCur() { cur = f(it.next()).iterator }

      def hasNext: Boolean = {
        while (!cur.hasNext) {
          if (!self.hasNext) return false
          nextCur()
        }
        true
      }

      def next(): U = (if (hasNext) cur else SharedIterator.empty).next()
    }
  }

  def existsExactly1(p: (T) => Boolean): Boolean = {
    var n: Int = 0
    while (it.hasNext)
      if (p(it.next())) {
        n += 1
        if (n > 1)
          return false
      }
    n == 1
  }

  def foreachBetween(f: (T) => Unit)(g: => Unit) {
    if (it.hasNext) {
      f(it.next())
      while (it.hasNext) {
        g
        f(it.next())
      }
    }
  }

  def pipe(pb: ProcessBuilder,
    printHeader: (String => Unit) => Unit,
    printElement: (String => Unit, T) => Unit,
    printFooter: (String => Unit) => Unit): (Iterator[String], Process) = {

    val command = pb.command().asScala.mkString(" ")

    val proc = pb.start()

    // Start a thread to print the process's stderr to ours
    new Thread("stderr reader for " + command) {
      override def run() {
        for (line <- Source.fromInputStream(proc.getErrorStream).getLines) {
          System.err.println(line)
        }
      }
    }.start()

    // Start a thread to feed the process input from our parent's iterator
    new Thread("stdin writer for " + command) {
      override def run() {
        val out = new PrintWriter(proc.getOutputStream)

        printHeader(out.println)
        it.foreach(x => printElement(out.println, x))
        printFooter(out.println)
        out.close()
      }
    }.start()

    // Return an iterator that read lines from the process's stdout
    (Source.fromInputStream(proc.getInputStream).getLines(), proc)
  }

}
