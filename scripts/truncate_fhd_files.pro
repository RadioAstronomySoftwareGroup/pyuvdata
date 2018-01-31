pro truncate_fhd_files
  ; dir = '/nfs/mwa-09/r1/djc/EoR2013/Aug23/fhd_nb_decon_March2016_small/'
  dir = '/nfs/mwa-04/r1/EoRuvfits/analysis/fhd_nb_Aug2017_savedbp_w_cable_w_digjump/'
  observation = '1061316296'
  outdir = dir+'data_for_tests/'
  
  if file_test(outdir, /directory) eq 0 then file_mkdir, outdir
  
  file_copy, dir+'metadata/'+observation+'_layout.sav', outdir+observation+'_layout.sav', /overwrite
  file_copy, dir+'metadata/'+observation+'_settings.txt', outdir+observation+'_settings.txt', /overwrite
  
  chan_range = [204-1,204+1]
  t0 = 4 ; Starting time so we don't include the first two seconds (which are flagged)
  nts_keep = 4 ; Will keep roughly 1/10 the times
  ntile_keep = 9
  restore,dir+'metadata/'+observation+'_obs.sav' ; Need the obs to get time bins
  restore,dir+'calibration/'+observation+'_cal.sav'
  restore,dir+'metadata/'+observation+'_params.sav'
  tmin_ind = (*obs.baseline_info).bin_offset[t0]
  tmax_ind = (*obs.baseline_info).bin_offset[t0+nts_keep] ; Keep fraction of the original baseline times
  blt_inds = where(((*obs.baseline_info).tile_a lt (ntile_keep+1)) and ((*obs.baseline_info).tile_b lt (ntile_keep+1)) $
    and (params.time ge params.time[tmin_ind]) and (params.time lt params.time[tmax_ind]))
  npol = 2
  pols=['XX','YY']
  
  ; Now for the hard part
  print,'Reorganizing obs structure'
  bin_offset = Lonarr(nts_keep)
  for t=1,(nts_keep-1) do bin_offset[t] = min(where(params.time[blt_inds] gt params.time[blt_inds[bin_offset[t-1]]]))
  new_baseline_info = structure_update(*obs.baseline_info, $
    tile_a=(*obs.baseline_info).tile_a[blt_inds], $
    tile_b=(*obs.baseline_info).tile_b[blt_inds], $
    bin_offset = bin_offset, $
    jdate = ((*obs.baseline_info).jdate)[t0:(t0+nts_keep-1)], $
    freq = ((*obs.baseline_info).freq)[chan_range[0]:chan_range[1]], $
    fbin_i = ((*obs.baseline_info).fbin_i)[chan_range[0]:chan_range[1]], $
    freq_use = ((*obs.baseline_info).freq_use)[chan_range[0]:chan_range[1]], $
    time_use = ((*obs.baseline_info).time_use)[t0:(t0+nts_keep-1)])
    
  obs_new = structure_update(obs, n_freq=chan_range[1]-chan_range[0]+1, $
    n_time = nts_keep, nf_vis = obs.nf_vis[chan_range[0]:chan_range[1]],$
    nbaselines = ntile_keep*(ntile_keep-1)/2)
  *obs_new.baseline_info = new_baseline_info
  *obs_new.vis_noise = (*obs.vis_noise)[*,chan_range[0]:chan_range[1]]
  obs=obs_new
  save,obs,filename=outdir+observation+'_obs.sav'
  
  print,'Reorganizing cal structure'
  gain = cal.gain
  convergence = cal.convergence
  auto_params = cal.auto_params
  for pol=0,1 do begin
    *gain[pol] = (*gain[pol])[chan_range[0]:chan_range[1],0:ntile_keep-1]
    *convergence[pol] = (*convergence[pol])[chan_range[0]:chan_range[1],0:ntile_keep-1]
    *auto_params[pol] = (*auto_params[pol])[*,0:ntile_keep-1]
  endfor
  cal_new = structure_update(cal, n_freq=chan_range[1]-chan_range[0]+1, $
    n_tile = ntile_keep, n_time = nts_keep,  uu = cal.uu[blt_inds], $
    vv = cal.vv[blt_inds], tile_a=cal.tile_a[blt_inds], tile_b=cal.tile_b[blt_inds], $
    tile_names = cal.tile_names[0:ntile_keep-1], bin_offset=bin_offset, $
    freq = cal.freq[chan_range[0]:chan_range[1]], gain = gain, $
    convergence = convergence, auto_params = auto_params, $
    amp_params = cal.amp_params[*,0:ntile_keep-1], $
    phase_params = cal.phase_params[*,0:ntile_keep-1], $
    mode_params = cal.mode_params[*,0:ntile_keep-1])
  cal=cal_new
  save,cal,filename=outdir+observation+'_cal.sav'
  
  ; params
  print,'Slicing params'
  params_new = {params,   uu:params.uu[blt_inds], $
    vv:params.vv[blt_inds], $
    ww:params.ww[blt_inds], $
    baseline_arr:params.baseline_arr[blt_inds], $
    time:params.time[blt_inds]}
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

  for pol=0,1 do begin
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
