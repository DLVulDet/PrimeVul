#### Issue

Dear author, 

There are many data in train.jsonl or train_paired.jsonl not associated with a CWE type. Could you please add these missing CWE types? 

For example, the cwe type for the data entry below is CWE-415 according to https://nvd.nist.gov/vuln/detail/CVE-2021-40572. But it is missing in the data dictionary.

Thank you!

#### Example
```
{"func": "void gf_av1_reset_state(AV1State *state, Bool is_destroy)\n{\n\tGF_List *l1, *l2;\n\n\tif (state->frame_state.header_obus) {\n\t\twhile (gf_list_count(state->frame_state.header_obus)) {\n\t\t\tGF_AV1_OBUArrayEntry *a = (GF_AV1_OBUArrayEntry*)gf_list_pop_back(state->frame_state.header_obus);\n\t\t\tif (a->obu) gf_free(a->obu);\n\t\t\tgf_free(a);\n\t\t}\n\t}\n\n\tif (state->frame_state.frame_obus) {\n\t\twhile (gf_list_count(state->frame_state.frame_obus)) {\n\t\t\tGF_AV1_OBUArrayEntry *a = (GF_AV1_OBUArrayEntry*)gf_list_pop_back(state->frame_state.frame_obus);\n\t\t\tif (a->obu) gf_free(a->obu);\n\t\t\tgf_free(a);\n\t\t}\n\t}\n\tl1 = state->frame_state.frame_obus;\n\tl2 = state->frame_state.header_obus;\n\tmemset(&state->frame_state, 0, sizeof(AV1StateFrame));\n\tstate->frame_state.is_first_frame = GF_TRUE;\n\n\tif (is_destroy) {\n\t\tgf_list_del(l1);\n\t\tgf_list_del(l2);\n\t\tif (state->bs) {\n\t\t\tu32 size;\n\t\t\tgf_bs_get_content_no_truncate(state->bs, &state->frame_obus, &size, &state->frame_obus_alloc);\n\t\t\tgf_bs_del(state->bs);\n\t\t}\n\t\tstate->bs = NULL;\n\t}\n\telse {\n\t\tstate->frame_state.frame_obus = l1;\n\t\tstate->frame_state.header_obus = l2;\n\t\tif (state->bs)\n\t\t\tgf_bs_seek(state->bs, 0);\n\t}\n}", "project": "gpac", "hash": 45274239427546243078887835425651724873, "size": 41, "commit_id": "7bb1b4a4dd23c885f9db9f577dfe79ecc5433109", "message": "fixed #1893", "target": 0, "dataset": "other", "idx": 217933}
```

