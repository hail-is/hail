#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "hail/allocators.hpp"
#include "hail/format.hpp"
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

static PyObject *HailType_str(HailTypeObject *self);
static PyTypeObject HailType = {
  .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "_hail.HailType",
  .tp_basicsize = sizeof(HailTypeObject),
  .tp_itemsize = 0, /* itemsize */
  .tp_str = (reprfunc)HailType_str,
  .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
  .tp_doc = "Hail type superclass",
  .tp_new = PyType_GenericNew,
};

struct TVoidObject : public HailTypeObject {};
struct TBoolObject : public HailTypeObject {};
struct TInt32Object : public HailTypeObject {};
struct TInt64Object : public HailTypeObject {};
struct TFloat32Object : public HailTypeObject {};
struct TFloat64Object : public HailTypeObject {};
struct TStrObject : public HailTypeObject {};

static PyObject *HailType_NewTVoid(PyTypeObject *subtype, PyObject *args, PyObject *kwds);
static PyObject *HailType_NewTBool(PyTypeObject *subtype, PyObject *args, PyObject *kwds);
static PyObject *HailType_NewTInt32(PyTypeObject *subtype, PyObject *args, PyObject *kwds);
static PyObject *HailType_NewTInt64(PyTypeObject *subtype, PyObject *args, PyObject *kwds);
static PyObject *HailType_NewTFloat32(PyTypeObject *subtype, PyObject *args, PyObject *kwds);
static PyObject *HailType_NewTFloat64(PyTypeObject *subtype, PyObject *args, PyObject *kwds);
static PyObject *HailType_NewTStr(PyTypeObject *subtype, PyObject *args, PyObject *kwds);

static PyTypeObject TVoid = {
  .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "_hail._tvoid",
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "Hail type for boolean (true or false) values",
  .tp_base = &HailType,
  .tp_new = HailType_NewTVoid,
};

static PyTypeObject TBool = {
  .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "_hail._tbool",
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "Hail type for boolean (true or false) values",
  .tp_base = &HailType,
  .tp_new = HailType_NewTBool,
};

static PyTypeObject TInt32 = {
  .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "_hail._tint32",
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "Hail type for 32 bit signed integers",
  .tp_base = &HailType,
  .tp_new = HailType_NewTInt32,
};

static PyTypeObject TInt64 = {
  .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "_hail._tint64",
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "Hail type for 64 bit signed integers",
  .tp_base = &HailType,
  .tp_new = HailType_NewTInt64,
};

static PyTypeObject TFloat32 = {
  .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "_hail._tfloat32",
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "Hail type for IEEE-754 32 bit floating point numbers",
  .tp_base = &HailType,
  .tp_new = HailType_NewTFloat32,
};

static PyTypeObject TFloat64 = {
  .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "_hail._tfloat64",
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "Hail type for IEEE-754 64 bit floating point numbers",
  .tp_base = &HailType,
  .tp_new = HailType_NewTFloat64,
};

static PyTypeObject TStr = {
  .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "_hail._tstr",
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "Hail type for strings",
  .tp_base = &HailType,
  .tp_new = HailType_NewTStr,
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

PyMODINIT_FUNC
PyInit__hail(void)
{
  PyObject *mod = PyModule_Create(&hail_types_module);
  PyObject *tvoid{}, *tbool{}, *tint32{}, *tint64{}, *tfloat32{}, *tfloat64{};
  PyObject *tstr{};
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
      || PyType_Ready(&TStr) < 0)
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
  if (!tvoid || !tbool || !tint32 || !tint64 || !tfloat32 || !tfloat64 || !tstr) {
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

  return mod;

error:
  Py_XDECREF(tvoid);
  Py_XDECREF(tbool);
  Py_XDECREF(tint32);
  Py_XDECREF(tint64);
  Py_XDECREF(tfloat32);
  Py_XDECREF(tfloat64);
  Py_XDECREF(tstr);
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
HailType_New(PyTypeObject *type, const hail::Type * hail_type)
{
  HailTypeObject *self = (HailTypeObject *)type->tp_alloc(type, 0);
  if (self == nullptr) return nullptr;
  self->type = hail_type;
  return (PyObject *)self;
}

static PyObject *
HailType_NewTVoid(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  return HailType_New(type, pytcx.tvoid);
}

static PyObject *
HailType_NewTBool(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  return HailType_New(type, pytcx.tbool);
}

static PyObject *
HailType_NewTInt32(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  return HailType_New(type, pytcx.tint32);
}

static PyObject *
HailType_NewTInt64(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  return HailType_New(type, pytcx.tint64);
}

static PyObject *
HailType_NewTFloat32(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  return HailType_New(type, pytcx.tfloat32);
}

static PyObject *
HailType_NewTFloat64(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  return HailType_New(type, pytcx.tfloat64);
}

static PyObject *
HailType_NewTStr(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  return HailType_New(type, pytcx.tstr);
}

} // namespace
