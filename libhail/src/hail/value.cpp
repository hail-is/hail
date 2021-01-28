#include <cstring>

#include <hail/format.hpp>
#include <hail/value.hpp>

namespace hail {

StrValue
Value::make_str(const PStr *ptype, std::shared_ptr<ArenaAllocator> region, size_t size) {
  StrData *p = (StrData *)region->allocate(4, 4 + size);
  p->size = size;
  return StrValue(ptype, std::move(region), p);
}

ArrayValue
Value::make_array(const PArray *ptype, std::shared_ptr<ArenaAllocator> region, size_t size) {
  ArrayData *p = (ArrayData *)region->allocate(alignof(ArrayData), sizeof(ArrayData));
  p->size = size;
  p->missing_bits = (char *)region->allocate(1, (size + 7) >> 3);
  p->elements = (char *)region->allocate(ptype->elements_alignment,
					 size * ptype->element_stride);
  return ArrayValue(ptype, std::move(region), p);
}

TupleValue
Value::make_tuple(const PTuple *ptype, std::shared_ptr<ArenaAllocator> region) {
  char *p = (char *)region->allocate(ptype->alignment, ptype->byte_size);
  return TupleValue(ptype, std::move(region), p);
}

Value
Value::load(const PType *ptype, std::shared_ptr<ArenaAllocator> region, void *p) {
  switch (ptype->tag) {
  case PType::Tag::BOOL:
    return Value(cast<PBool>(ptype), *(bool *)p);
  case PType::Tag::INT32:
    return Value(cast<PInt32>(ptype), *(int32_t *)p);
  case PType::Tag::INT64:
    return Value(cast<PInt64>(ptype), *(int64_t *)p);
  case PType::Tag::FLOAT32:
    return Value(cast<PFloat32>(ptype), *(float *)p);
  case PType::Tag::FLOAT64:
    return Value(cast<PFloat64>(ptype), *(double *)p);
  case PType::Tag::STR:
    return Value(cast<PStr>(ptype), std::move(region), *(void **)p);
  case PType::Tag::ARRAY:
    return Value(cast<PArray>(ptype), std::move(region), *(void **)p);
  case PType::Tag::TUPLE:
      return Value(cast<PTuple>(ptype), std::move(region), p);
  default:
    abort();
  }
}

void
Value::store(void *p, const Value &value) {
  auto ptype = value.get_ptype();
  switch (ptype->tag) {
  case PType::Tag::BOOL:
    *(bool *)p = value.u.b;
    break;
  case PType::Tag::INT32:
    *(int32_t *)p = value.u.i32;
    break;
  case PType::Tag::INT64:
    *(int64_t *)p = value.u.i64;
    break;
  case PType::Tag::FLOAT32:
    *(float *)p = value.u.f;
    break;
  case PType::Tag::FLOAT64:
    *(float *)p = value.u.d;
    break;
  case PType::Tag::STR:
    *(void **)p = value.u.p;
    break;
  case PType::Tag::ARRAY:
    *(void **)p = value.u.p;
    break;
  case PType::Tag::TUPLE:
    {
      auto ptuple = cast<PTuple>(ptype);
      TupleValue src = value.as_tuple();
      TupleValue dest(ptuple, value.region, (char *)p);
      size_t n = ptuple->element_ptypes.size();
      memcpy(p, src.p, (n + 7) / 8);
      for (size_t i = 0; i < n; ++i) {
	if (src.get_element_present(i))
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
  const PType *ptype = value.get_ptype();
  switch (ptype->tag) {
  case PType::Tag::VOID:
    format(s, "()");
    break;
  case PType::Tag::BOOL:
    format(s, value.as_bool());
    break;
  case PType::Tag::INT32:
    format(s, value.as_int32());
    break;
  case PType::Tag::INT64:
    format(s, value.as_int64());
    break;
  case PType::Tag::FLOAT32:
    format(s, value.as_float32());
    break;
  case PType::Tag::FLOAT64:
    format(s, value.as_float64());
    break;
  case PType::Tag::STR:
    {
      auto t = value.as_str();
      s.write(t.get_data(), t.get_size());
    }
    break;
  case PType::Tag::ARRAY:
    {
      auto a = value.as_array();
      s.putc('(');
      for (int i = 0; i < a.get_size(); ++i) {
	if (a.get_element_present(i))
	  format(s, a.get_element(i));
	else
	  format(s, "null");
      }
      s.putc(')');
    }
    break;
  case PType::Tag::TUPLE:
    {
      auto t = value.as_tuple();
      s.putc('(');
      for (int i = 0; i < t.get_size(); ++i) {
	if (i)
	  s.puts(", ");
	if (t.get_element_present(i))
	  format(s, t.get_element(i));
	else
	  format(s, "null");
      }
      s.putc(')');
    }
    break;
  default:
    abort();
    
  }
}

}
