# Grep

This module doesn't actually have any specific link to sequence data, but is a convenience provided to those in the statistical genetics community who often work on enormous text files like VCFs.  `grep` mimics the basic functionality of Unix grep, but does so harnessing the parallel environment of spark to speed up a query by orders of magnitude.

Command line options:
 - `-i | --input <file>` -- Grep this file (the file should be in hadoop, or this module will run no more quickly than Unix `grep`)
 - `-r | --regex <regex>` -- Write regex here.  Should follow [java regex convention](https://docs.oracle.com/javase/7/docs/api/java/util/regex/Pattern.html)
 - `-o | --output <file>` -- Write matches to this file instead of printing to standard out. (optional)
 - `-s | --stop` -- Write only one match (will NOT write the first match)
 