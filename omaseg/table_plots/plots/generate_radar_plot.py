from omaseg.table_plots.plots.utils import read_results_table
from omaseg.dataset_utils.bodyparts_labelmaps import anatomical_systems
from omaseg.table_plots.plots.plot_functions import generate_radar_plot


if __name__ == "__main__":
    table_path = '/mnt/hdda/murong/22k/results/compare_totalseg_omaseg_p005/group_by_omaseg_inout/filtered_unreliable_and_limited_fov/dice_compare_table.xlsx'  #TODO: 
    column_name = 'all median'  #TODO: 
    omaseg_dice, totalseg_dice = read_results_table(table_path, column_name)
        
    generate_radar_plot(
        model1_scores=totalseg_dice,
        model2_scores=omaseg_dice,
        model1_name='TotalSeg',
        model2_name='OMASeg',
        output_path = "/mnt/hdda/murong/22k/plots/per-structure/radar_plot_unreliable_limitedFOV_all_median_dice.png",  #TODO: 
        title="Structure-wise Median Dice Score Comparison (Full Dataset)",  #TODO: 
        circle_positions=[0.5, 0.6, 0.8, 0.9, 1.0],  #TODO: 
        system_groups=anatomical_systems
    )
