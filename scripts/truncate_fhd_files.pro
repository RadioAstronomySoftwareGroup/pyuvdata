pro truncate_fhd_files
  dir = '/nfs/mwa-09/r1/djc/EoR2013/Aug23/fhd_nb_decon_March2016_small/'
  observation = '1061316296'
  outdir = dir+'data_for_tests/'

  chan_range = [196-1,196+1]
  nts_keep = 5 ; Will keep roughly 1/10 the times
  restore,dir+'metadata/'+observation+'_obs.sav' ; Need the obs to get time bins
  nblts_keep = (*obs.baseline_info).bin_offset[nts_keep] ; Keep fraction of the original baseline times
  npol = 2
  pols=['XX','YY']

  restore,dir+'vis_data/'+observation+'_flags.sav'  ; Flag file
  for pol=0,1 do begin
    *flag_arr[pol] = (*flag_arr[pol])[chan_range[0]:chan_range[1],0:(nblts_keep-1)]
  endfor
  save,flag_arr,filename=outdir+observation+'_flags.sav'
  undefine,flag_arr

  for pol=0,1 do begin
    restore,dir+'vis_data/'+observation+'_vis_'+pols[pol]+'.sav' ; Dirty data
    *vis_ptr = (*vis_ptr)[chan_range[0]:chan_range[1],0:(nblts_keep-1)]
    save,vis_ptr,filename=outdir+observation+'_vis_'+pols[pol]+'.sav'
    restore,dir+'vis_data/'+observation+'_vis_model_'+pols[pol]+'.sav' ; Model data
    *vis_model_ptr = (*vis_model_ptr)[chan_range[0]:chan_range[1],0:(nblts_keep-1)]
    save,vis_model_ptr,filename=outdir+observation+'_vis_model_'+pols[pol]+'.sav'
  endfor

  ; Now for the hard part
  new_baseline_info = structure_update(*obs.baseline_info, $
    tile_a=(*obs.baseline_info).tile_a[0:(nblts_keep-1)], $
    tile_b=(*obs.baseline_info).tile_b[0:(nblts_keep-1)], $
    bin_offset = ((*obs.baseline_info).bin_offset)[0:(nts_keep-1)], $
    jdate = ((*obs.baseline_info).jdate)[0:(nts_keep-1)], $
    freq = ((*obs.baseline_info).freq)[chan_range[0]:chan_range[1]], $
    fbin_i = ((*obs.baseline_info).fbin_i)[chan_range[0]:chan_range[1]], $
    freq_use = ((*obs.baseline_info).freq_use)[chan_range[0]:chan_range[1]], $
    time_use = ((*obs.baseline_info).time_use)[0:(nts_keep-1)])

  obs_new = structure_update(obs, n_freq=chan_range[1]-chan_range[0]+1, $
    n_time = nts_keep, nf_vis = obs.nf_vis[chan_range[0]:chan_range[1]])
  *obs_new.baseline_info = new_baseline_info
  *obs_new.vis_noise = (*obs.vis_noise)[*,chan_range[0]:chan_range[1]]
  obs=obs_new
  save,obs,filename=outdir+observation+'_obs.sav'

  ; params
  restore,dir+'metadata/'+observation+'_params.sav'
  params_new = {params,   uu:params.uu[0:(nblts_keep-1)], $
    vv:params.vv[0:(nblts_keep-1)], $
    ww:params.ww[0:(nblts_keep-1)], $
    baseline_arr:params.baseline_arr[0:(nblts_keep-1)], $
    time:params.time[0:(nblts_keep-1)]}
  params = params_new
  save,params,filename=outdir+observation+'_params.sav'

end
