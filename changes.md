
 - filtervariants and filtersamples have been reworked. Much like the annotate commands, you will need to specify `filtersamples expr` or `filtervariants intervals`. See the updated [filtering documentation](https://github.com/broadinstitute/hail/blob/master/docs/Filtering.md) for details.

 - filtervariants now supports filtering on .variant_list with lines
   of the form contig:pos:ref:alt1,alt2,...altN.
   
 - added new functions in expr language on arrays/sets: `filter`, `map`, `min`, `max`, `sum`.
 
 - added new function in expr language `str(x)` which gives a string representation of a thing (comma-delimited for arrays which can be changed with `arr.mkString(delim)`, and produces json for structs)  

 - `isNotMissing` renamed to `isDefined`.  `isMissing` and
   `isDefined` are now single-argument functions and can be used
   with structs, e.g., `isMissing(x)` instead of `x.isMissing`.

 - Fixed precedence bug in parser.  You can now write `!g.isHet`
   instead of `!(g.isNet)`.

 - count: `nLocalSamples` is gone.  `nSamples` reports the number of
   samples currently in the dataset.
