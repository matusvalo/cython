cimport cython
from ..StringIOTree cimport StringIOTree


cdef class AbstractUtilityCode:
    pass


cdef class UtilityCodeBase(AbstractUtilityCode):
    cpdef format_code(self, code_string, replace_empty_lines=*)


cdef class UtilityCode(UtilityCodeBase):
    cdef public object name
    cdef public object proto
    cdef public object export
    cdef public object impl
    cdef public object init
    cdef public object cleanup
    cdef object proto_block
    cdef object init_block
    cdef readonly object module_state_decls
    cdef readonly object module_state_traverse
    cdef readonly object module_state_clear
    cdef public object requires
    cdef dict _cache
    cdef list specialize_list
    cdef public object file
    cdef readonly tuple _parts_tuple
    cdef list shared_utility_functions

    cpdef none_or_sub(self, s, context)
    # TODO - Signature not compatible with previous declaration
    #@cython.final
    #cdef bint _put_code_section(self, writer, code_type: str) except -1


@cython.final
cdef class FunctionState:
    cdef set names_taken
    cdef public object owner
    cdef public object scope

    cdef public object error_label
    cdef public size_t label_counter
    cdef public set labels_used
    cdef public object return_label
    cdef public object continue_label
    cdef public object break_label
    cdef readonly list yield_labels

    cdef public object return_from_error_cleanup_label # not used in __init__ ?

    cdef public object exc_vars
    cdef public object current_except
    cdef public bint can_trace
    cdef public bint gil_owned

    cdef list[tuple] temps_allocated
    cdef dict[tuple, tuple] temps_free
    cdef dict[object, tuple] temps_used_type
    cdef set zombie_temps
    cdef size_t temp_counter
    cdef list[set[tuple]] collect_temps_stack

    cdef readonly object closure_temps
    cdef bint should_declare_error_indicator
    cdef public bint uses_error_indicator
    cdef public bint error_without_exception

    cdef public bint needs_refnanny

    cpdef new_label(self, name=*)
    cpdef tuple get_loop_labels(self)
    cpdef set_loop_labels(self, labels)
    cpdef tuple get_all_labels(self)
    cpdef set_all_labels(self, labels)
    cpdef start_collecting_temps(self)
    cpdef stop_collecting_temps(self)

    cpdef list[tuple] temps_in_use(self)

@cython.final
cdef class NumConst:
    cdef readonly object cname
    cdef readonly object value
    cdef readonly object py_type
    cdef readonly object value_code

@cython.final
cdef class PyObjectConst:
    cdef readonly object cname
    cdef readonly object type


@cython.final
cdef class StringConst:
    cdef readonly object cname
    cdef readonly object text
    cdef readonly object escaped_value
    cdef readonly dict[tuple, PyStringConst] py_strings
    cdef public bint c_used

    cpdef PyStringConst get_py_string_const(self, encoding, identifier=*)


@cython.final
cdef class PyStringConst:
    cdef readonly object cname
    cdef readonly object encoding
    cdef readonly bint is_unicode
    cdef readonly bint intern


cdef class GlobalState:
    cdef public tuple module_pos
    cdef public dict directives
    cdef readonly list filename_list
    cdef readonly set utility_codes
    cdef readonly dict parts
    cdef readonly list shared_utility_functions
    cdef dict filename_table
    cdef dict input_file_contents
    cdef dict declared_cnames
    cdef bint in_utility_code_generation
    cdef object code_config
    cdef object common_utility_include_dir
    cdef public object module_node

    cdef dict const_cnames_used
    cdef dict[object, StringConst] string_const_index
    cdef dict dedup_const_index
    cdef dict pyunicode_ptr_const_index
    cdef list codeobject_constants
    cdef dict[tuple, object] num_const_index
    cdef list[PyObjectConst] arg_default_constants
    cdef dict const_array_counters
    cdef dict cached_cmethods
    cdef set initialised_constants

    cdef object rootwriter

    cdef StringConst get_string_const(self, text, bint c_used=*)
    cdef PyStringConst get_py_string_const(self, text, identifier=*)
    cdef str get_py_codeobj_const(self, node)
    cdef StringConst new_string_const(self, text, byte_string)
    cdef NumConst new_num_const(self, str value, str py_type, value_code=*)
    cdef str new_string_const_cname(self, bytes_value)
    cdef str unique_const_cname(self, str format_str)
    cpdef str new_const_cname(self, str prefix=*, str value=*)
    cdef str new_array_const_cname(self, str prefix)
    cdef str get_cached_unbound_method(self, type_cname, method_name)
    cpdef str get_py_const(self, prefix, dedup_key=*)
    cdef NumConst get_int_const(self, str str_value, bint longness=*)
    cdef NumConst get_float_const(self, str str_value, str value_code)

#def funccontext_property(name):

cdef class CCodeWriter(object):
    cdef readonly StringIOTree buffer
    cdef readonly list pyclass_stack
    cdef readonly GlobalState globalstate
    cdef readonly FunctionState funcstate
    cdef object code_config
    cdef tuple last_pos
    cdef tuple last_marked_pos
    cdef Py_ssize_t level
    cdef public Py_ssize_t call_level  # debug-only, see Nodes.py
    cdef bint bol

    @cython.final
    cdef void handle_refnanny(self, tp, bint nanny=*)
    cpdef write(self, s)
    cdef _write_lines(self, s)
    cpdef _write_to_buffer(self, s)
    cdef put_multilines(self, code)
    cpdef put(self, code)
    cpdef put_safe(self, code)
    cpdef putln(self, code=*, bint safe=*)
    cdef emit_marker(self)
    cdef _build_marker(self, tuple pos)
    cdef increase_indent(self)
    cdef decrease_indent(self)
    cdef indent(self)


cdef class PyrexCodeWriter:
    cdef readonly object f
    cdef readonly Py_ssize_t level


cdef class PyxCodeWriter:
    cdef public StringIOTree buffer
    cdef public object context
    cdef object encoding
    cdef Py_ssize_t level
    cdef Py_ssize_t original_level
    cdef dict _insertion_points
