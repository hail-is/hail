#include <cstring>

#include <hail/format.hpp>
#include <hail/value.hpp>

namespace hail {

StrValue
Value::make_str(const VStr *vtype, std::shared_ptr<ArenaAllocator> region, size_t size) {
  StrData *p = (StrData *)region->allocate(4, 4 + size);
  p->size = size;
  return StrValue(vtype, std::move(region), p);
}

ArrayValue
Value::make_array(const VArray *vtype, std::shared_ptr<ArenaAllocator> region, size_t size) {
  ArrayData *p = (ArrayData *)region->allocate(alignof(ArrayData), sizeof(ArrayData));
  p->size = size;
  p->missing_bits = (char *)region->allocate(1, (size + 7) >> 3);
  p->elements = (char *)region->allocate(vtype->elements_alignment,
					 size * vtype->element_stride);
  return ArrayValue(vtype, std::move(region), p);
}

TupleValue
Value::make_tuple(const VTuple *vtype, std::shared_ptr<ArenaAllocator> region) {
  char *p = (char *)region->allocate(vtype->alignment, vtype->byte_size);
  return TupleValue(vtype, std::move(region), p);
}

Value
Value::load(const VType *vtype, std::shared_ptr<ArenaAllocator> region, void *p) {
  switch (vtype->tag) {
  case VType::Tag::BOOL:
    return Value(cast<VBool>(vtype), *(bool *)p);
  case VType::Tag::INT32:
    return Value(cast<VInt32>(vtype), *(int32_t *)p);
  case VType::Tag::INT64:
    return Value(cast<VInt64>(vtype), *(int64_t *)p);
  case VType::Tag::FLOAT32:
    return Value(cast<VFloat32>(vtype), *(float *)p);
  case VType::Tag::FLOAT64:
    return Value(cast<VFloat64>(vtype), *(double *)p);
  case VType::Tag::STR:
    return Value(cast<VStr>(vtype), std::move(region), *(void **)p);
  case VType::Tag::ARRAY:
    return Value(cast<VArray>(vtype), std::move(region), *(void **)p);
  case VType::Tag::TUPLE:
      return Value(cast<VTuple>(vtype), std::move(region), p);
  default:
    abort();
  }
}

void
Value::store(void *p, const Value &value) {
  auto vtype = value.vtype;
  switch (vtype->tag) {
  case VType::Tag::BOOL:
    *(bool *)p = value.u.b;
    break;
  case VType::Tag::INT32:
    *(int32_t *)p = value.u.i32;
    break;
  case VType::Tag::INT64:
    *(int64_t *)p = value.u.i64;
    break;
  case VType::Tag::FLOAT32:
    *(float *)p = value.u.f;
    break;
  case VType::Tag::FLOAT64:
    *(double *)p = value.u.d;
    break;
  case VType::Tag::STR:
    *(void **)p = value.u.p;
    break;
  case VType::Tag::ARRAY:
    *(void **)p = value.u.p;
    break;
  case VType::Tag::TUPLE:
    {
      auto ptuple = cast<VTuple>(vtype);
      TupleValue src = value.as_tuple();
      TupleValue dest(ptuple, value.region, (char *)p);
      size_t n = ptuple->element_vtypes.size();
      memcpy(p, src.p, (n + 7) / 8);
      for (size_t i = 0; i < n; ++i) {
	if (!src.get_element_missing(i))
	  dest.set_element(i, src.get_element(i));
      }
    }
    break;
  default:
    abort();
  }
}



void
format1(FormatStream &s, const Value &value) {
  if (value.missing) {
    format(s, "null");
    return;
  }

  const VType *vtype = value.vtype;
  switch (vtype->tag) {
  case VType::Tag::VOID:
    format(s, "()");
    break;
  case VType::Tag::BOOL:
    format(s, value.as_bool());
    break;
  case VType::Tag::INT32:
    format(s, value.as_int32());
    break;
  case VType::Tag::INT64:
    format(s, value.as_int64());
    break;
  case VType::Tag::FLOAT32:
    format(s, value.as_float32());
    break;
  case VType::Tag::FLOAT64:
    format(s, value.as_float64());
    break;
  case VType::Tag::STR:
    {
      auto t = value.as_str();
      s.write(t.get_data(), t.get_size());
    }
    break;
  case VType::Tag::ARRAY:
    {
      auto a = value.as_array();
      s.putc('[');
      for (int i = 0; i < a.get_size(); ++i) {
	if (i)
	  s.puts(", ");
	format(s, a.get_element(i));
      }
      s.putc(']');
    }
    break;
  case VType::Tag::TUPLE:
    {
      auto t = value.as_tuple();
      s.putc('(');
      for (int i = 0; i < t.get_size(); ++i) {
	if (i)
	  s.puts(", ");
	format(s, t.get_element(i));
      }
      s.putc(')');
    }
    break;
  default:
    abort();
    
  }
}

}
