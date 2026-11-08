/* MicroPython binding for stuck_llm.
 *
 *   import stuck_llm
 *   m = stuck_llm.Model(open("/data/llm/model.bin", "rb").read(),
 *                       open("/data/llm/tokenizer.bin", "rb").read())
 *   m.set_sampler(0.9, 0.9, 1234)
 *   m.begin("Where am I?")
 *   while True:
 *       print(m.next_piece(), end="")
 *
 * The model bytes object is kept referenced by the Model and used in place,
 * so the weights live wherever MicroPython put that bytes object. On the
 * badge that is the PSRAM-backed heap. The KV cache and sampler scratch go
 * through heap_caps_malloc into PSRAM on ESP32, plain malloc elsewhere, which
 * also lets this module build into the unix port for testing.
 */

#include <stdlib.h>
#include <string.h>

#include "py/obj.h"
#include "py/runtime.h"

#include "stuck_llm.h"

#ifdef ESP_PLATFORM
#include "esp_heap_caps.h"
static void *psram_alloc(size_t n)
{
    void *p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    return p ? p : malloc(n);
}
static void psram_free(void *p) { heap_caps_free(p); }
#endif

typedef struct {
    mp_obj_base_t base;
    mp_obj_t model_bytes;   /* keeps the weights alive */
    stuck_llm_t *m;
} stuck_model_obj_t;

static void check(int rc, const char *what)
{
    if (rc < 0) mp_raise_msg_varg(&mp_type_RuntimeError, MP_ERROR_TEXT("stuck_llm %s failed (%d)"), what, rc);
}

static mp_obj_t stuck_model_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args)
{
    mp_arg_check_num(n_args, n_kw, 2, 2, false);
#ifdef ESP_PLATFORM
    stuck_set_big_allocator(psram_alloc, psram_free);
#endif
    mp_buffer_info_t model, tok;
    mp_get_buffer_raise(args[0], &model, MP_BUFFER_READ);
    mp_get_buffer_raise(args[1], &tok, MP_BUFFER_READ);
    if (((uintptr_t)model.buf & 3) != 0) {
        mp_raise_ValueError(MP_ERROR_TEXT("model bytes must be 4-byte aligned; use bytes, not a slice"));
    }
    stuck_model_obj_t *self = mp_obj_malloc_with_finaliser(stuck_model_obj_t, type);
    self->model_bytes = args[0];
    self->m = NULL;
    check(stuck_init(&self->m, model.buf, model.len), "init");
    int rc = stuck_load_tokenizer(self->m, tok.buf, tok.len);
    if (rc < 0) { stuck_free(self->m); self->m = NULL; check(rc, "tokenizer"); }
    return MP_OBJ_FROM_PTR(self);
}

static stuck_llm_t *get(mp_obj_t self_in)
{
    stuck_model_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if (!self->m) mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("model closed"));
    return self->m;
}

static mp_obj_t stuck_model_set_sampler(size_t n_args, const mp_obj_t *args)
{
    float temp = mp_obj_get_float(args[1]);
    float topp = n_args > 2 ? mp_obj_get_float(args[2]) : 0.9f;
    uint64_t seed = n_args > 3 ? (uint64_t)mp_obj_get_int(args[3]) : 0;
    stuck_set_sampler(get(args[0]), temp, topp, seed);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(stuck_model_set_sampler_obj, 2, 4, stuck_model_set_sampler);

static mp_obj_t stuck_model_begin(size_t n_args, const mp_obj_t *args)
{
    const char *prompt = NULL;
    if (n_args > 1 && args[1] != mp_const_none) prompt = mp_obj_str_get_str(args[1]);
    check(stuck_monologue_begin(get(args[0]), prompt), "begin");
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(stuck_model_begin_obj, 1, 2, stuck_model_begin);

static mp_obj_t stuck_model_next_piece(mp_obj_t self_in)
{
    const char *piece = stuck_next_piece(get(self_in));
    if (!piece) mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("generation failed"));
    size_t n = strlen(piece);
    /* A lone raw byte that is not printable ASCII would make an invalid str;
     * the panel font cannot show it anyway, so drop it. */
    if (n == 1 && (unsigned char)piece[0] >= 0x80) return MP_OBJ_NEW_QSTR(MP_QSTR_);
    if (n == 1 && (unsigned char)piece[0] < 0x20 && piece[0] != '\n') return MP_OBJ_NEW_QSTR(MP_QSTR_);
    return mp_obj_new_str(piece, n);
}
static MP_DEFINE_CONST_FUN_OBJ_1(stuck_model_next_piece_obj, stuck_model_next_piece);

static mp_obj_t stuck_model_pos(mp_obj_t self_in) { return MP_OBJ_NEW_SMALL_INT(stuck_pos(get(self_in))); }
static MP_DEFINE_CONST_FUN_OBJ_1(stuck_model_pos_obj, stuck_model_pos);

static mp_obj_t stuck_model_info(mp_obj_t self_in)
{
    stuck_llm_t *m = get(self_in);
    mp_obj_t items[5] = {
        MP_OBJ_NEW_SMALL_INT(stuck_dim(m)), MP_OBJ_NEW_SMALL_INT(stuck_n_layers(m)),
        MP_OBJ_NEW_SMALL_INT(stuck_vocab_size(m)), MP_OBJ_NEW_SMALL_INT(stuck_seq_len(m)),
        mp_obj_new_int_from_uint(stuck_runtime_bytes(m)),
    };
    return mp_obj_new_tuple(5, items);   /* (dim, layers, vocab, seq_len, runtime_bytes) */
}
static MP_DEFINE_CONST_FUN_OBJ_1(stuck_model_info_obj, stuck_model_info);

static mp_obj_t stuck_model_close(mp_obj_t self_in)
{
    stuck_model_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if (self->m) { stuck_free(self->m); self->m = NULL; }
    self->model_bytes = mp_const_none;
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(stuck_model_close_obj, stuck_model_close);

static const mp_rom_map_elem_t stuck_model_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_set_sampler), MP_ROM_PTR(&stuck_model_set_sampler_obj) },
    { MP_ROM_QSTR(MP_QSTR_begin), MP_ROM_PTR(&stuck_model_begin_obj) },
    { MP_ROM_QSTR(MP_QSTR_next_piece), MP_ROM_PTR(&stuck_model_next_piece_obj) },
    { MP_ROM_QSTR(MP_QSTR_pos), MP_ROM_PTR(&stuck_model_pos_obj) },
    { MP_ROM_QSTR(MP_QSTR_info), MP_ROM_PTR(&stuck_model_info_obj) },
    { MP_ROM_QSTR(MP_QSTR_close), MP_ROM_PTR(&stuck_model_close_obj) },
    { MP_ROM_QSTR(MP_QSTR___del__), MP_ROM_PTR(&stuck_model_close_obj) },
};
static MP_DEFINE_CONST_DICT(stuck_model_locals_dict, stuck_model_locals_dict_table);

MP_DEFINE_CONST_OBJ_TYPE(
    stuck_model_type,
    MP_QSTR_Model,
    MP_TYPE_FLAG_NONE,
    make_new, stuck_model_make_new,
    locals_dict, &stuck_model_locals_dict
);

static const mp_rom_map_elem_t stuck_llm_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_stuck_llm) },
    { MP_ROM_QSTR(MP_QSTR_Model), MP_ROM_PTR(&stuck_model_type) },
};
static MP_DEFINE_CONST_DICT(stuck_llm_globals, stuck_llm_globals_table);

const mp_obj_module_t stuck_llm_module = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *)&stuck_llm_globals,
};

MP_REGISTER_MODULE(MP_QSTR_stuck_llm, stuck_llm_module);
