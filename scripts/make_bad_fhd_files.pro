pro make_bad_fhd_files, good_data_file, good_layout_file, good_flag_file

  ;; obs comes out of the data file, need to mess it up there.
  if n_elements(good_data_file) eq 0 then good_data_file = '1061316296_vis_XX.sav'
  data_root = cgRootName(good_data_file, directory=directory, extension=extension)
  data_start = (strsplit(data_root, 'vis',/extract))[0]
  data_end = 'vis' + (strsplit(data_root, 'vis',/extract))[1]
  bad_data_file = directory + data_start + 'broken_' + data_end + '.' + extension

  restore, good_data_file
  obs.instrument = 'foo'
  obs.n_time = 1
  obs.nbaselines = obs.nbaselines / 2.
  obs.obsra = obs.obsra - 10.

  save, file = bad_data_file, obs, vis_ptr

  bad_obs_loc_data_file = directory + data_start + 'bad_obs_loc_' + data_end + '.' + extension
  restore, good_data_file

  obs.lat = obs.lat + 10
  obs.lon = obs.lon + 10

  nants = n_elements((*obs.baseline_info).tile_names)
  (*obs.baseline_info).tile_names = string(indgen(nants))

  save, file = bad_obs_loc_data_file, obs, vis_ptr


  if n_elements(good_layout_file) eq 0 then good_layout_file = '1061316296_layout.sav'
  layout_root = cgRootName(good_layout_file, directory=directory, extension=extension)
  layout_start = (strsplit(layout_root, 'layout',/extract))[0]
  layout_end = 'layout'
  bad_layout_file = directory + layout_start + 'broken_' + layout_end + '.' + extension

  restore, good_layout_file

  layout.coordinate_frame = '????'

  nants = n_elements(layout.antenna_names)
  layout = create_struct(layout, 'diameters', fltarr(nants) + 5.)

  layout = create_struct(layout, 'foo', 'bar')

  save, file = bad_layout_file, layout

  good_arr_center_layout_file = directory + layout_start + 'fixed_arr_center_' + layout_end + '.' + extension
  restore, good_layout_file

  layout.array_center = [-2559454.07880307d,  5095372.14368305d, -2849057.18534633d]
  save, file = good_arr_center_layout_file, layout


  if n_elements(good_flag_file) eq 0 then good_flag_file = '1061316296_flags.sav'
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
