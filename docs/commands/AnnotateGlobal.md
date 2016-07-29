# `annotateglobal`

This module is a supercommand that contains three submodules:

Name | Docs | Description
:-:  | :-: | ---
`annotateglobal expr` | [**\[link\]**](AnnotateGlobalExpr.md) | Generate global annotations using the Hail expr language, including the ability to aggregate sample and variant statistics
`annotateglobal list` | [**\[link\]**](AnnotateGlobalList.md) | Read a file to global annotations as an `Array[String]` or `Set[String]`.
`annotateglobal table` | [**\[link\]**](AnnotateGlobalTable.md) | Read a file to global annotations as an `Array[Struct]` using Hail's table parsing module.