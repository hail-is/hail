#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "hail/allocators.hpp"
#include "hail/format.hpp"
#include "hail/tunion.hpp"
#include "hail/type.hpp"

namespace {

hail::HeapAllocator alloc;
hail::TypeContext pytcx{alloc};

/* methods exposed by this module */
static PyMethodDef HailTypesMethods[] = {
  { nullptr, nullptr, 0, nullptr }
};


struct HailTypeObject {
  PyObject_HEAD
  const hail::Type *type;
};

/* basic hail type methods */
static PyObject *HailType_str(HailTypeObject *self);
static PyObject *HailType_repr(HailTypeObject *self);
static PyObject *HailType_richcmp(HailTypeObject *self, PyObject *other, int op);
static Py_hash_t HailType_hash(HailTypeObject *self);

static PyTypeObject HailType = {
  .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "_hail.HailType",
  .tp_basicsize = sizeof(HailTypeObject),
  .tp_itemsize = 0, /* itemsize */
  .tp_repr = (reprfunc)HailType_repr,
  .tp_hash = (hashfunc)HailType_hash,
  .tp_str = (reprfunc)HailType_str,
  .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
  .tp_doc = "Hail type superclass",
  .tp_richcompare = (richcmpfunc)HailType_richcmp,
  .tp_new = PyType_GenericNew,
};

static PyObject *TVoid_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds);
static PyTypeObject TVoid = {
  .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "_hail._tvoid",
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "Hail type for boolean (true or false) values",
  .tp_base = &HailType,
  .tp_new = TVoid_new,
};

static PyObject *TBool_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds);
static PyTypeObject TBool = {
  .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "_hail._tbool",
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "Hail type for boolean (true or false) values",
  .tp_base = &HailType,
  .tp_new = TBool_new,
};

static PyObject *TInt32_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds);
static PyTypeObject TInt32 = {
  .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "_hail._tint32",
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "Hail type for 32 bit signed integers",
  .tp_base = &HailType,
  .tp_new = TInt32_new,
};

static PyObject *TInt64_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds);
static PyTypeObject TInt64 = {
  .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "_hail._tint64",
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "Hail type for 64 bit signed integers",
  .tp_base = &HailType,
  .tp_new = TInt64_new,
};

static PyObject *TFloat32_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds);
static PyTypeObject TFloat32 = {
  .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "_hail._tfloat32",
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "Hail type for IEEE-754 32 bit floating point numbers",
  .tp_base = &HailType,
  .tp_new = TFloat32_new,
};

static PyObject *TFloat64_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds);
static PyTypeObject TFloat64 = {
  .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "_hail._tfloat64",
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "Hail type for IEEE-754 64 bit floating point numbers",
  .tp_base = &HailType,
  .tp_new = TFloat64_new,
};

static PyObject *TStr_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds);
static PyTypeObject TStr = {
  .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "_hail._tstr",
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "Hail type for strings",
  .tp_base = &HailType,
  .tp_new = TStr_new,
};

static PyObject *TCall_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds);
static PyTypeObject TCall = {
  .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "_hail._tcall",
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "Hail type for genotypes",
  .tp_base = &HailType,
  .tp_new = TCall_new,
};

static int TArray_init(HailTypeObject *self, PyObject *args, PyObject *kwds);
static PyObject *TArray_get_element_type(HailTypeObject *self, void *closure);
static PyGetSetDef TArray_getset[] = {
    {"element_type", (getter) TArray_get_element_type, nullptr, "element type", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}  /* Sentinel */
};
static PyTypeObject TArray = {
  .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "_hail.tarray",
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "Hail type for arrays",
  .tp_getset = TArray_getset,
  .tp_base = &HailType,
  .tp_init = (initproc)TArray_init,
};

static int TStream_init(HailTypeObject *self, PyObject *args, PyObject *kwds);
static PyObject *TStream_get_element_type(HailTypeObject *self, void *closure);
static PyGetSetDef TStream_getset[] = {
    {"element_type", (getter) TStream_get_element_type, nullptr, "element type", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}  /* Sentinel */
};
static PyTypeObject TStream = {
  .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "_hail.tstream",
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "Hail type for streams",
  .tp_getset = TStream_getset,
  .tp_base = &HailType,
  .tp_init = (initproc)TStream_init,
};

static int TTuple_init(HailTypeObject *self, PyObject *args, PyObject *kwds);
static Py_ssize_t TTuple_len(HailTypeObject *self);
static PyObject *TTuple_getitem(HailTypeObject *self, Py_ssize_t index);
static PySequenceMethods TTuple_SeqMethods = {
  .sq_length = (lenfunc)TTuple_len,
  .sq_item = (ssizeargfunc)TTuple_getitem,
};
// FIXME possibly add a faster iterator
static PyTypeObject TTuple = {
  .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "_hail.ttuple",
  .tp_as_sequence = &TTuple_SeqMethods,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "Hail type for tuples",
  .tp_base = &HailType,
  .tp_init = (initproc)TTuple_init,
};

PyDoc_STRVAR(_hail_module_doc, "hail native methods/interface");
static struct PyModuleDef hail_types_module = {
  PyModuleDef_HEAD_INIT,
  "_hail",   /* name of module */
  _hail_module_doc, /* module documentation, may be NULL */
  -1,       /* size of per-interpreter state of the module,
               or -1 if the module keeps state in global variables. */
  HailTypesMethods, /* methods go here */
  nullptr, /* m_slots */
  nullptr, /* m_traverse */
  nullptr, /* m_clear */
  nullptr, /* m_free */
};

// public object singletons
static PyObject *tvoid, *tbool, *tint32, *tint64, *tfloat32, *tfloat64, *tstr, *tcall;

PyMODINIT_FUNC
PyInit__hail(void)
{
  PyObject *mod = PyModule_Create(&hail_types_module);
  if (mod == nullptr) {
    goto error;
  }

  if (PyType_Ready(&HailType) < 0
      || PyType_Ready(&TVoid) < 0
      || PyType_Ready(&TBool) < 0
      || PyType_Ready(&TInt32) < 0
      || PyType_Ready(&TInt64) < 0
      || PyType_Ready(&TFloat32) < 0
      || PyType_Ready(&TFloat64) < 0
      || PyType_Ready(&TStr) < 0
      || PyType_Ready(&TCall) < 0
      || PyType_Ready(&TArray) < 0
      || PyType_Ready(&TStream) < 0
      || PyType_Ready(&TTuple) < 0)
  {
    goto error;
  }

  tvoid = TVoid.tp_new(&TVoid, NULL, NULL);
  tbool = TBool.tp_new(&TBool, NULL, NULL);
  tint32 = TInt32.tp_new(&TInt32, NULL, NULL);
  tint64 = TInt64.tp_new(&TInt64, NULL, NULL);
  tfloat32 = TFloat32.tp_new(&TFloat32, NULL, NULL);
  tfloat64 = TFloat64.tp_new(&TFloat64, NULL, NULL);
  tstr = TStr.tp_new(&TStr, NULL, NULL);
  tcall = TCall.tp_new(&TCall, NULL, NULL);
  if (!tvoid || !tbool || !tint32 || !tint64 || !tfloat32 || !tfloat64 || !tstr || !tcall) {
    goto error;
  }

  Py_INCREF(tvoid);
  if (PyModule_AddObject(mod, "tvoid", tvoid) < 0) {
    Py_DECREF(tvoid);
    goto error;
  }

  Py_INCREF(tbool);
  if (PyModule_AddObject(mod, "tbool", tbool) < 0) {
    Py_DECREF(tbool);
    goto error;
  }

  Py_INCREF(tint32);
  if (PyModule_AddObject(mod, "tint32", tint32) < 0) {
    Py_DECREF(tint32);
    goto error;
  }

  Py_INCREF(tint64);
  if (PyModule_AddObject(mod, "tint64", tint64) < 0) {
    Py_DECREF(tint64);
    goto error;
  }

  Py_INCREF(tfloat32);
  if (PyModule_AddObject(mod, "tfloat32", tfloat32) < 0) {
    Py_DECREF(tfloat32);
    goto error;
  }

  Py_INCREF(tfloat64);
  if (PyModule_AddObject(mod, "tfloat64", tfloat64) < 0) {
    Py_DECREF(tfloat64);
    goto error;
  }

  Py_INCREF(tstr);
  if (PyModule_AddObject(mod, "tstr", tstr) < 0) {
    Py_DECREF(tstr);
    goto error;
  }

  Py_INCREF(tcall);
  if (PyModule_AddObject(mod, "tcall", tcall) < 0) {
    Py_DECREF(tcall);
    goto error;
  }

  Py_INCREF(&TArray);
  if (PyModule_AddObject(mod, "tarray", (PyObject *)&TArray) < 0) {
    Py_DECREF(&TArray);
    goto error;
  }

  Py_INCREF(&TStream);
  if (PyModule_AddObject(mod, "tstream", (PyObject *)&TStream) < 0) {
    Py_DECREF(&TStream);
    goto error;
  }

  Py_INCREF(&TTuple);
  if (PyModule_AddObject(mod, "ttuple", (PyObject *)&TTuple) < 0) {
    Py_DECREF(&TStream);
    goto error;
  }

  return mod;

error:
  Py_XDECREF(tvoid);
  Py_XDECREF(tbool);
  Py_XDECREF(tint32);
  Py_XDECREF(tint64);
  Py_XDECREF(tfloat32);
  Py_XDECREF(tfloat64);
  Py_XDECREF(tstr);
  Py_XDECREF(tcall);
  Py_XDECREF(mod);
  return nullptr;
}

static PyObject *
HailType_str(HailTypeObject *self)
{
  if (self->type == nullptr) {
    Py_RETURN_NONE;
  }

  hail::StringFormatStream sfs;
  format1(sfs, self->type);
  std::string str = sfs.get_contents();
  return PyUnicode_FromString(str.c_str());
}

static PyObject *
HailType_repr(HailTypeObject *self)
{
  PyObject *str = HailType_str(self);

  if (!PyUnicode_Check(str)) {
    // this is None (or NULL), return it
    return str;
  }

  // constants, intermidiate, and return strings
  PyObject *substr{}, *repstr{}, *tmpstr{}, *retstr{};
  substr = PyUnicode_FromString("'");
  if (substr == nullptr) goto hailtype_repr_out;
  repstr = PyUnicode_FromString("\\'");
  if (repstr == nullptr) goto hailtype_repr_out;
  tmpstr = PyUnicode_Replace(str, substr, repstr, -1);
  if (tmpstr == nullptr) goto hailtype_repr_out;
  retstr = PyUnicode_FromFormat("dtype('%U')", tmpstr);

hailtype_repr_out:
  Py_DECREF(str);
  Py_XDECREF(substr);
  Py_XDECREF(repstr);
  Py_XDECREF(tmpstr);
  return retstr;
}

static Py_hash_t
HailType_hash(HailTypeObject *self)
{
  /* bottom 3 or 4 bits are likely to be 0; rotate y by 4 to avoid
     excessive hash collisions for dicts and sets */
  uintptr_t y = (size_t)self->type;
  y = (y >> 4) | (y << (8 * SIZEOF_VOID_P - 4));
  Py_hash_t hash = y;
  return hash == -1 ? -2 : hash;
}

static PyObject *
HailType_richcmp(HailTypeObject *self, PyObject *other, int op)
{
  if (op != Py_EQ && op != Py_NE) {
    Py_RETURN_NOTIMPLEMENTED;
  }
  switch (PyObject_IsInstance(other, (PyObject*)&HailType)) {
    case -1: return nullptr;
    case 1: break;
    default: Py_RETURN_FALSE;
  }
  Py_RETURN_RICHCOMPARE(self->type, ((HailTypeObject*)other)->type, op);
}

static PyObject *
HailType_New(PyTypeObject *type, const hail::Type * hail_type)
{
  HailTypeObject *self = reinterpret_cast<HailTypeObject *>(type->tp_alloc(type, 0));
  if (self == nullptr) return nullptr;
  self->type = hail_type;
  return reinterpret_cast<PyObject *>(self);
}

#define RETURN_TVOID do { Py_INCREF(tvoid); return tvoid; } while (0)
#define RETURN_TBOOL do { Py_INCREF(tbool); return tbool; } while (0)
#define RETURN_TINT32 do { Py_INCREF(tint32); return tint32; } while (0)
#define RETURN_TINT64 do { Py_INCREF(tint64); return tint64; } while (0)
#define RETURN_TFLOAT32 do { Py_INCREF(tfloat32); return tfloat32; } while (0)
#define RETURN_TFLOAT64 do { Py_INCREF(tfloat64); return tfloat64; } while (0)
#define RETURN_TSTR do { Py_INCREF(tstr); return tstr; } while (0)

// utility to construct the appropriate HailType python subclass
// from a hail::Type *
static PyObject *
HailType_from_Type(const hail::Type *type)
{
  PyObject *obj{};
  switch (type->tag) {
    case hail::Type::Tag::VOID:
      RETURN_TVOID;
    case hail::Type::Tag::BOOL:
      RETURN_TBOOL;
    case hail::Type::Tag::INT32:
      RETURN_TINT32;
    case hail::Type::Tag::INT64:
      RETURN_TINT64;
    case hail::Type::Tag::FLOAT32:
      RETURN_TFLOAT32;
    case hail::Type::Tag::FLOAT64:
      RETURN_TFLOAT64;
    case hail::Type::Tag::STR:
      RETURN_TSTR;
    case hail::Type::Tag::ARRAY:
      obj = TArray.tp_new(&TArray, NULL, NULL);
      break;
    case hail::Type::Tag::STREAM:
      obj = TStream.tp_new(&TStream, NULL, NULL);
      break;
    case hail::Type::Tag::TUPLE:
      obj = TTuple.tp_new(&TTuple, NULL, NULL);
      break;
    default:
      abort();
  }
  if (obj) {
    ((HailTypeObject *)obj)->type = type;
  }
  return obj;
}

static PyObject *
TVoid_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  return HailType_New(type, pytcx.tvoid);
}

static PyObject *
TBool_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  return HailType_New(type, pytcx.tbool);
}

static PyObject *
TInt32_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  return HailType_New(type, pytcx.tint32);
}

static PyObject *
TInt64_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  return HailType_New(type, pytcx.tint64);
}

static PyObject *
TFloat32_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  return HailType_New(type, pytcx.tfloat32);
}

static PyObject *
TFloat64_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  return HailType_New(type, pytcx.tfloat64);
}

static PyObject *
TStr_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  return HailType_New(type, pytcx.tstr);
}

static PyObject *
TCall_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  return HailType_New(type, pytcx.tcall);
}

static int
TArray_init(HailTypeObject *self, PyObject *args, PyObject *kwds)
{
  const char *kw_names[] = {"element_type", nullptr};
  HailTypeObject *py_element_type{};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!:__init__", (char **)kw_names,
                                   &HailType, &py_element_type)) {
    return -1;
  }
  const hail::TArray *type = pytcx.tarray(py_element_type->type);
  self->type = type;
  return 0;
}

static PyObject *
TArray_get_element_type(HailTypeObject *self, void *closure)
{
  return HailType_from_Type(hail::cast<hail::TArray>(self->type)->element_type);
}

static int
TStream_init(HailTypeObject *self, PyObject *args, PyObject *kwds)
{
  const char *kw_names[] = {"element_type", nullptr};
  HailTypeObject *py_element_type{};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!:__init__", (char **)kw_names,
                                   &HailType, &py_element_type)) {
    return -1;
  }
  const hail::TStream *type = pytcx.tstream(py_element_type->type);
  self->type = type;
  return 0;
}

static PyObject *
TStream_get_element_type(HailTypeObject *self, void *closure)
{
  return HailType_from_Type(hail::cast<hail::TStream>(self->type)->element_type);
}

static int
TTuple_init(HailTypeObject *self, PyObject *args, PyObject *kwds)
{
  std::vector<const hail::Type *> element_types;
  Py_ssize_t n_args = PyTuple_Size(args);
  for (auto i = 0; i < n_args; i++) {
    PyObject *item = PyTuple_GET_ITEM(args, i);
    switch (PyObject_IsInstance(item, (PyObject*)&HailType)) {
      case -1: return -1;
      case 0:
        PyErr_Format(PyExc_TypeError,
                     "expected argument %d of __init__() to be %.200s, "
                     "not %.200s", i + 1, HailType.tp_name,
                     _PyType_Name(Py_TYPE(item)));
        return -1;
      case 1: break;
      default: abort();
    }
    element_types.push_back(reinterpret_cast<HailTypeObject *>(item)->type);
  }
  self->type = pytcx.ttuple(element_types);
  return 0;
}

static Py_ssize_t
TTuple_len(HailTypeObject *self)
{
  return hail::cast<hail::TTuple>(self->type)->element_types.size();
}

static PyObject *TTuple_getitem(HailTypeObject *self, Py_ssize_t index)
{
  auto types = hail::cast<hail::TTuple>(self->type)->element_types;
  if (index >= (Py_ssize_t)types.size()) {
    PyErr_SetString(PyExc_IndexError, "tuple index out of range");
    return nullptr;
  }
  return HailType_from_Type(types[index]);
}

} // namespace
