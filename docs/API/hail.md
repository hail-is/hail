This document is a draft of version 1 of the AMP T2D Hail REST API.

# getStats
## Input Parameters
* `passback`        : String. Returned as is in the response JSON to facilitate debug. Optional. Default: None.
* `md_version`      : String. Indicates metadata version. Optional. Default: mdv1 (GoT2D data).
* `api_version`     : Integer. Indicates API version to synchronize clients. Required.
* `phenotype`       : String. Encodes the trait of interest. List of available covariates are published through the metadata API. Optional. Default: t2d. Currently precomputed p-values are returned and this is ignored.
* `covariates`      : Array of covariate objects. Used in conditional analysis. List of available covariates are published through the metadata API. Optional. Default: No covariates. Currently precomputed p-values are returned and this is ignored.
* `variant_filters` : Array of filter objects that are logically AND'ed. Required. Each filter statement is comprised of a list of key-value pairs that define the search/filter operation. The filter statements contain the following keys that specify the formula to filter on:
  1. `operand`      : String. Associated value specifies the property/parameter used in the formula.
  2. `operator`     : String. Associated value specifies formula operator (eq, gte, gt, lte, lt).
  3. `value`        : String. Associated value specifies the value used in the formula.
  4. `operand_type` : String. Specifies the type of the property (integer, float, string).
* `limit`           : Integer. Maximum number of results to be returned. Optional. Default: Implementation specific. Currently, 10K with a hard limit of 10K.
* `count`           : Boolean. True if only the number of results is desired to be returned rather than the results themselves. Optional. Default: false.

**Example input:**
```json
{
  "passback"        : "example",
  "md_version"      : "mdv1",
  "api_version"     : 1,
  "phenotype"       : "t2d",
  "covariates"      : [
                    {"type": "principal component", "name": "C1"},
                    {"type": "phenotype", "name": "BMI"},
                    {"type": "variant", "chrom": "20", "pos": 9012, "ref": "T", "alt": "G"}
                  ],
  "variant_filters" : [
                        {"operand": "chrom", "operator": "eq", "value": "20", "operand_type": "string"},
                        {"operand": "position", "operator": "gte", "value": 1234, "operand_type": "integer"},
                        {"operand": "position", "operator": "lte", "value": 5678, "operand_type": "integer"}
                      ],
  "limit"           : 50,
  "count"           : false,
  "sort_by"         : ["chrom", "pos"]
}
```

## Output Parameters
#### Statistical Parameters
* `stats`  : Array of objects. Maps variants to their (floating-point) association statistics.
* `count`  : Integer.  Number of results available for query.  No larger than `limit`, if specified.

#### Debug Parameters
* `passback`      : String. Contains the `passback` value given in the original request.
* `is_error`      : Boolean. True if the operation errored out due to bad input or an internal issue.
* `error_message` : String. Indicates the cause of failure if `is_error` is `true`.

**Example output:**
```json
{
    "is_error"  : false,
    "passback"  : "example",
    "stats"     : [
                    {"chrom": "20", "pos": 1235, "ref": "A", "alt": "C", "p-value": 0.0035},
                    {"chrom": "20", "pos": 2974, "ref": "G", "alt": "TCA", "p-value": 0.1},
                    {"chrom": "20", "pos": 3811, "ref": "TTA", "alt": "", "p-value": 0.9}
                  ]
}
```

# getLDValues
Not yet implemented.

# getMetadata
TBD
