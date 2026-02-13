package is.hail.asm4s

import is.hail.annotations.Region
import is.hail.io.{InputBuffer, OutputBuffer}

package object implicits {
  implicit def valueToRichCodeInputBuffer(in: Value[InputBuffer]): RichCodeInputBuffer =
    new RichCodeInputBuffer(in)

  implicit def valueToRichCodeOutputBuffer(out: Value[OutputBuffer]): RichCodeOutputBuffer =
    new RichCodeOutputBuffer(out)

  implicit def toRichCodeIterator[T](it: Code[Iterator[T]]): RichCodeIterator[T] =
    new RichCodeIterator[T](it)

  implicit def valueToRichCodeIterator[T](it: Value[Iterator[T]]): RichCodeIterator[T] =
    new RichCodeIterator[T](it)

  implicit def codeToRichCodeRegion(region: Code[Region]): RichCodeRegion =
    new RichCodeRegion(region)

  implicit def valueToRichCodeRegion(region: Value[Region]): RichCodeRegion =
    new RichCodeRegion(region)
}
