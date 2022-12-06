// RUN: hail-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

func.func @apply_type_mismatch(%x: i64) -> i64 {
  // expected-note @below {{prior use here}}
  %res = cps.callcc %ret : i32 {
    // expected-error @below {{use of value '%ret' expects different type than prior uses: '!cps.cont<i64>' vs '!cps.cont<i32>'}}
    cps.apply %ret(%x) : i64
  }
  return %res
}

// -----

func.func @apply_type_mismatch2(%x: i64) -> i64 {
  %res = cps.callcc %ret : i32 {
    // expected-error @below {{argument types match continuation type}}
    "cps.apply"(%ret, %x) : (!cps.cont<i32>, i64) -> ()
  }
  return %res : i32
}

// -----

func.func @apply_arg_number(%x: i32) -> i32 {
  %res = cps.callcc %ret : i32 {
    // expected-error @below {{argument types match continuation type}}
    "cps.apply"(%ret, %x, %x) : (!cps.cont<i32>, i32, i32) -> ()
  }
  return %res : i32
}

// -----

func.func @defcont_no_blocks(%x: i32) {
  cps.callcc %ret {
    // expected-error @below {{failed to verify constraint: region with 1 blocks}}
    %cont = "cps.cont"() ({
    }) : () -> (!cps.cont<i32>)
    cps.apply %cont(%x) : i32
  }
  return
}

// -----

func.func @defcont_arg_types(%x: i32) {
  cps.callcc %ret {
    // expected-error @below {{type mismatch between 0th block arg and continuation type}}
    %cont = "cps.cont"() ({
      ^bb0(%arg: i64):
        cps.apply %ret
    }) : () -> (!cps.cont<i32>)
    cps.apply %cont(%x) : i32
  }
  return
}
