<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Description:

The filter genotypes module has only one function, the `expr` function, so it is not broken into submodules.  

Removed genotypes will be set to missing.  Use the [Hail expression language](reference.html#HailExpressionLanguage) to supply a boolean expression involving the following exposed data structures:

Exposed Name | Description
:-: | ---
 `g`  | genotype
 `s`  | sample
 `sa` | sample annotation
 `v`  | variant
 `va` | variant annotation
 `global` | global annotation

   
For more information about these exposed objects and how to use them, see the documentation on [representation](reference.html#Representation) and the [Hail expression language](reference.html#HailExpressionLanguage).
</div>