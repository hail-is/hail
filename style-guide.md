# Scala Style Guide
## IntelliJ settings
### Automatic

The easiest way to comply with the style guide is to load the IntelliJ
`code_style.xml`. You can load the xml file using the `Manage ...` button on the
`Editor > Code Style` page. After loading it, select it from the drop-down list.

### Manual

Our style differs slightly from the IntelliJ Scala plugin defaults.
Make the following changes:

 - Turn off Preferences > Editor > Code Style > Syntax > Other > Enforce procedure syntax for methods with Unit return type.

 - Turn on Preferences > Editor > Code Style > Scala > Spaces > Other > Insert whitespaces in simple one line blocks.

 - Turn off Preferences > Editor > Code Style > Scala > Wrapping and Braces > Align when multiline in all categories.

## Guide

 - Prefer
   ```scala
   def foo() { ... }
   ```
   to
   ```scala
   def foo(): Unit = { ... }
   ```
   In IntelliJ, turn off Preferences > Editor > Code Style > Syntax > Other > Enforce procedure syntax for methods with Unit return type.

 - Prefix mutable data structures with mutable.  That is, prefer
   ```scala
   import scala.collection.mutable
   ... mutable.ArrayBuilder[Byte] ...
   ```
   to
   ```scala
   import scala.collection.mutable.ArrayBuilder
   ... ArrayBuilder[Byte] ...
   ```

 - Prefer `None` to `Option.empty`. Prefer `Some(_)` to `Option(_)` when the argument will never be `null`.

 - Use require, assert and ensure liberally to check preconditions, conditions and post-conditions.  Define a validate member to check object invariants and call where suitable.

 - In IntelliJ, turn on Preferences > Editor > Code Style > Scala > Spaces > Other > Insert whitespaces in simple one line blocks.

 - In IntelliJ, turn off Preferences > Editor > Code Style > Scala > Wrapping and Braces > Align when multiline in all categories.  This helps the code from migrating too far to the right.
 
