# Changelog
main
----
* Updated SKA LOW station coordinates to revision 4 and config_create functions ([MR31](https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels/-/merge_requests/31))

0.2.0
----
* Add support for a wider varient of casa calibration tables, G, B, Df, K and Kcross ([MR30](https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels/-/merge_requests/30))
* Add the support of dealing with weights of different diameter antenna ([MR27](https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels/-/merge_requests/27))
* Add support for casa calibration tables with a row for each time and antenna combination ([MR26](https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels/-/merge_requests/26))

0.1.3
----
* Add functions to read GainTable from CASA table ([MR21](https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels/-/merge_requests/21))
* add function export_skymodel_to_text, needed to add the option of calibration with DP3 in rascil ([MR23](https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels/-/merge_requests/23))
* Update GainTable to read different receptor frames as inputs ([MR22](https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels/-/merge_requests/22))
* get_direction_time_location and calculate_visibility_hourangles allow for user-defined time inputs ([MR20](https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels/-/merge_requests/20))
* Bug fix in PointingTable.pointingtable_acc.nrec ([MR20](https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels/-/merge_requests/20))
* Bug fix when simulating with more than 2045 frequency channels (([MR12](https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels/-/merge_requests/12)))

0.1.2
-----
* Add copy functions for SkyComponent and SkyModel (([MR18](https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels/-/merge_requests/18)))

0.1.1
-----
* From RASCIL, added, create_gaintable_from_visibility, create_pointingtable_from_visibility,
  create_griddata_from_image, create_convolutionfunction_from_image, create_flagtable_from_visibility
  ([MR17](https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels/-/merge_requests/17))
* Added create_image from RASCIL ([MR15](https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels/-/merge_requests/15))

0.1.0
-----
* Added create_visibility and various create_configuration functions from RASCIL ([MR13](https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels/-/merge_requests/13))
* Move class methods of classes inheriting from Dataset into the accessor classes ([MR11](https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels/-/merge_requests/11))
* Restructured the repository ([MR10](https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels/-/merge_requests/10))
* Moved Image() into its own file and added unit tests for it ([MR8](https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels/-/merge_requests/8))
* Documentation improvements and updates ([MR9](https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels/-/merge_requests/9), [MR7](https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels/-/merge_requests/7), [MR4](https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels/-/merge_requests/4))
* Migrated data models from RASCIL ([MR3](https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels/-/merge_requests/3))
