pro truncate_fhd_files, npol=npol, chan_range=chan_range, time_range=time_range, $
    tile_range = tile_range, outdir_str=outdir_str, galaxy_model = galaxy_model, $
    diffuse_model = diffuse_model
  ; dir = '/nfs/mwa-09/r1/djc/EoR2013/Aug23/fhd_nb_decon_March2016_small/'
  dir = '/nfs/mwa-04/r1/EoRuvfits/analysis/fhd_nb_Aug2017_savedbp_w_cable_w_digjump/'
  observation = '1061316296'
  if n_elements(outdir_str) eq 0 then outdir_str = ''
  outdir = dir+'data_for_tests' + outdir_str + '/'

  if file_test(outdir, /directory) eq 0 then file_mkdir, outdir

  file_copy, dir+'metadata/'+observation+'_layout.sav', outdir+observation+'_layout.sav', /overwrite
  file_copy, dir+'metadata/'+observation+'_settings.txt', outdir+observation+'_settings.txt', /overwrite

  if n_elements(chan_range) eq 0 then chan_range = [204-1,204+1]
  ; Starting time so we don't include the first two seconds (which are flagged)
  if n_elements(time_range) eq 0 then time_range =[4, 7] ; Will keep roughly 1/10 the times
  if n_elements(tile_range) eq 0 then tile_range = [0, 8]
  n_times = time_range[1] - time_range[0] + 1
  n_tiles = tile_range[1] - tile_range[0] + 1

  restore,dir+'metadata/'+observation+'_obs.sav' ; Need the obs to get time bins
  restore,dir+'calibration/'+observation+'_cal.sav'
  restore,dir+'metadata/'+observation+'_params.sav'

  tmin_ind = (*obs.baseline_info).bin_offset[time_range[0]]
  tmax_ind = (*obs.baseline_info).bin_offset[time_range[1]] ; Keep fraction of the original baseline times
  blt_inds = where(((*obs.baseline_info).tile_a ge (tile_range[0])) and ((*obs.baseline_info).tile_b ge (tile_range[0])) $
    and ((*obs.baseline_info).tile_a le (tile_range[1])) and ((*obs.baseline_info).tile_b le (tile_range[1])) $
    and (params.time ge params.time[tmin_ind]) and (params.time le params.time[tmax_ind]))
  if n_elements(npol) eq 0 then npol = 2
  if npol lt 0 or npol gt 2 then message, 'npol can only be 1 or 2'
  if npol eq 1 then pols=['XX'] else pols=['XX','YY']


  ; Now for the hard part
  print,'Reorganizing obs structure'
  bin_offset = Lonarr(n_times)
  for t=1,(n_times-1) do bin_offset[t] = min(where(params.time[blt_inds] gt params.time[blt_inds[bin_offset[t-1]]]))
  new_baseline_info = structure_update(*obs.baseline_info, $
    tile_a=(*obs.baseline_info).tile_a[blt_inds], $
    tile_b=(*obs.baseline_info).tile_b[blt_inds], $
    bin_offset = bin_offset, $
    jdate = ((*obs.baseline_info).jdate)[time_range[0]:time_range[1]], $
    freq = ((*obs.baseline_info).freq)[chan_range[0]:chan_range[1]], $
    fbin_i = ((*obs.baseline_info).fbin_i)[chan_range[0]:chan_range[1]], $
    freq_use = ((*obs.baseline_info).freq_use)[chan_range[0]:chan_range[1]], $
    time_use = ((*obs.baseline_info).time_use)[time_range[0]:time_range[1]], $
    tile_use = ((*obs.baseline_info).tile_use)[tile_range[0]:tile_range[1]])

  obs_new = structure_update(obs, n_freq=chan_range[1]-chan_range[0]+1, $
    n_time = n_times, nf_vis = obs.nf_vis[chan_range[0]:chan_range[1]],$
    nbaselines = n_tiles*(n_tiles-1)/2, n_pol = npol)
  *obs_new.baseline_info = new_baseline_info
  *obs_new.vis_noise = (*obs.vis_noise)[*,chan_range[0]:chan_range[1]]
  obs=obs_new
  save,obs,filename=outdir+observation+'_obs.sav'

  print,'Reorganizing cal structure'
  gain = cal.gain[0:npol-1]
  gain_residual = cal.gain_residual[0:npol-1]
  convergence = cal.convergence[0:npol-1]
  auto_params = cal.auto_params[0:npol-1]
  for pol=0,npol-1 do begin
    *gain[pol] = (*gain[pol])[chan_range[0]:chan_range[1],tile_range[0]:tile_range[1]]
    *gain_residual[pol] = (*gain_residual[pol])[chan_range[0]:chan_range[1],tile_range[0]:tile_range[1]]
    *convergence[pol] = (*convergence[pol])[chan_range[0]:chan_range[1],tile_range[0]:tile_range[1]]
    *auto_params[pol] = (*auto_params[pol])[*,tile_range[0]:tile_range[1]]
  endfor
  nsrc_keep = 15
  new_skymodel = structure_update(cal.skymodel, n_sources = nsrc_keep, $
    source_list = cal.skymodel.source_list[0:nsrc_keep-1])
  if keyword_set(galaxy_model) then begin
    new_skymodel.galaxy_model = 1
  endif
  if n_elements(diffuse_model) gt 0 then begin
    new_skymodel.diffuse_model = diffuse_model
  endif
  cal_new = structure_update(cal, n_freq=chan_range[1]-chan_range[0]+1, $
    n_tile = n_tiles, n_time = n_times, n_pol = npol, $
    uu = cal.uu[blt_inds], $
    vv = cal.vv[blt_inds], tile_a=cal.tile_a[blt_inds], tile_b=cal.tile_b[blt_inds], $
    tile_names = cal.tile_names[tile_range[0]:tile_range[1]], bin_offset=bin_offset, $
    freq = cal.freq[chan_range[0]:chan_range[1]], gain = gain, $
    gain_residual = gain_residual, $
    convergence = convergence, auto_params = auto_params, $
    amp_params = cal.amp_params[*,tile_range[0]:tile_range[1]], $
    phase_params = cal.phase_params[*,tile_range[0]:tile_range[1]], $
    mode_params = cal.mode_params[*,tile_range[0]:tile_range[1]], $
    skymodel=new_skymodel)
  cal=cal_new
  save,cal,filename=outdir+observation+'_cal.sav'

  ; params
  print,'Slicing params'
  params_new = structure_update(params, $
    uu=params.uu[blt_inds], $
    vv=params.vv[blt_inds], $
    ww=params.ww[blt_inds], $
    baseline_arr=params.baseline_arr[blt_inds], $
    time=params.time[blt_inds])
  params = params_new
  save,params,filename=outdir+observation+'_params.sav'

  restore,dir+'vis_data/'+observation+'_flags.sav'  ; Flag file
  if n_elements(flag_arr) gt 0 then begin
    for pol=0,1 do begin
      *flag_arr[pol] = (*flag_arr[pol])[chan_range[0]:chan_range[1],blt_inds]
    endfor
    save,flag_arr,filename=outdir+observation+'_flags.sav'
    undefine,flag_arr
  endif

  for pol=0, npol-1 do begin
    print,'Slicing pol '+pols[pol]
    restore,dir+'vis_data/'+observation+'_vis_'+pols[pol]+'.sav' ; Dirty data
    obs = obs_new
    *vis_ptr = (*vis_ptr)[chan_range[0]:chan_range[1],blt_inds]
    save,vis_ptr,obs,filename=outdir+observation+'_vis_'+pols[pol]+'.sav'
    restore,dir+'vis_data/'+observation+'_vis_model_'+pols[pol]+'.sav' ; Model data
    obs = obs_new
    *vis_model_ptr = (*vis_model_ptr)[chan_range[0]:chan_range[1],blt_inds]
    save,vis_model_ptr,obs,filename=outdir+observation+'_vis_model_'+pols[pol]+'.sav'
  endfor

end
