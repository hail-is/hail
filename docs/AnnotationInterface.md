```
class Path(a: Array[String])

// Might not always be Row.
type Annotation = Row

object Annotation {
 def structAnnotation(a: Array[Any]): Annotation

 def same(that: Annotation): Boolean
}

type Deleter = (Annotation) => Annotation
type Querier = (Annotation) => Any
type Inserter = (Annotation, Any) => Annotation

abstract class Signature {
 def typeOf: String
 def typeOf(p: Path): String

 def toExprType: expr.Type

 def delete(p: Path): (Signature, Deleter)
 def query(p: Path): Querier = {
   if (p.isEmpty)
     a => a
   else
     throw new IllegalArgumentException
 }

 /* if p = a.b.c, inserts at the end of a.b.
  
  There are two overwrite situations.  You are inserting a.b and it
  already exists (possibly with a different type).  You are
  inserting a.b.c and a.b exists and has non-struct type.  */
 def insert(p: Path, s: Signature, overwrite: Boolean): (Signature, Inserter)

 /* if p = a.b.c, inserts f after c in a.b. */
 def insertAfter(p: Path, f: String, s: Signature): (Signature, Inserter)
}

class StructSignature(m: Map[String, (Int, Signature)]) extends AnnotationSignature {
 def query(p: Path): Querier = {
   if (p.empty)
     a => a
   else {
     val f = p.head
     m.get(f) match {
       case Some((i, s)) =>
         q = s.query(p.tail)
         a => q(a(i))
       case None =>
         // f not a member of m
         throw new IllegalArgumentExcpetion
     }
   }
 }
}

// combine?
class SimpleSignature extends AnnotationSignature
class VCFSignature extends AnnotationSignature

object StructSignature {
 def empty: StructSignature
 ```