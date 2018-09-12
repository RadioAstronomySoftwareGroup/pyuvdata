pro make_bad_fhd_files, good_data_file, good_layout_file, good_flag_file

  ;; obs comes out of the data file, need to mess it up there.
  data_root = cgRootName(good_data_file, directory=directory, extension=extension)
  data_start = (strsplit(data_root, 'vis',/extract))[0]
  data_end = 'vis' + (strsplit(data_root, 'vis',/extract))[1]
  bad_data_file = directory + data_start + 'broken_' + data_end + '.' + extension

  restore, good_data_file
  obs.instrument = 'foo'
  obs.n_time = 1
  obs.nbaselines = obs.nbaselines / 2.
  (*obs.baseline_info).freq = [(*obs.baseline_info).freq, (*obs.baseline_info).freq + obs.freq_res*3]
  obs.obsra = obs.obsra - 10.

  nants = n_elements((*obs.baseline_info).tile_names)
  (*obs.baseline_info).tile_names = string(indgen(nants))

  save, file = bad_data_file, obs, vis_ptr


  layout_root = cgRootName(good_layout_file, directory=directory, extension=extension)
  layout_start = (strsplit(layout_root, 'layout',/extract))[0]
  layout_end = 'layout'
  bad_layout_file = directory + layout_start + 'broken_' + layout_end + '.' + extension

  restore, good_layout_file
  tags = strlowcase(tag_names(layout))
  wh_keep = where(tags ne 'array_center' and tags ne 'coordinate_frame', count_keep)
  if count_keep gt 0 then begin
    new_layout = create_struct(tags[wh_keep[0]], layout.(wh_keep[0]))
    for i=1, n_elements(tags_to_keep) - 1 do begin
      new_layout = create_struct(new_layout, tags[wh_keep[i]], layout.(wh_keep[i]))
    endfor
  endif

  nants = n_elements(layout.antenna_names)
  new_layout = create_struct(new_layout, 'diameters', fltarr(nants) + 5.)
  new_layout = create_struct(new_layout, 'foo', 'bar')
  layout=new_layout

  save, file = bad_layout_file, layout


  flag_root = cgRootName(good_flag_file, directory=directory, extension=extension)
  flag_start = (strsplit(layout_root, 'flags',/extract))[0]
  flag_end = 'flags'
  variant_flag_file = directory + flag_start + 'variant_' + flag_end + '.' + extension
  bad_flag_file = directory + flag_start + 'broken_' + flag_end + '.' + extension

  restore, good_flag_file

  vis_weights = flag_arr
  save, file = variant_flag_file, vis_weights

  foo = ''
  save, file = bad_flag_file, foo

end
