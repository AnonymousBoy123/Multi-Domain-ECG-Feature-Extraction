% 使用 matfile 逐步加载 EEG 数据
data1D_name = '../4_processed_data/KUL_1D.mat';
EEG_mat = matfile(data1D_name);  % 以 matfile 方式加载，避免内存溢出

% 设置输出目录
output_folder = '../EEG_TimeSeries_Plots/';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

sbnum = 16;  % 受试者数
trnum = 8;   % 试验数
fs = 128;    % 采样率 128Hz
trial_id = 1;  % 只可视化第一个 trial

for sb = 1:sbnum
    disp(['Processing Subject ' num2str(sb)]);  % 进度信息

    % **使用 matfile() 逐步读取 EEG 数据**
    eeg_data = squeeze(EEG_mat.EEG(sb, trial_id, :, :));  % 读取单个受试者的数据

    figure('Visible', 'off');  % 不显示图像，节省内存
    time_axis = (1:size(eeg_data,1)) / fs;  % 计算时间轴 (秒)

    hold on;
    for ch = 1:64
        plot(time_axis, eeg_data(:, ch) + ch * 10, 'k');  % 叠加偏移量, 避免重叠
    end
    hold off;

    title(['Subject ' num2str(sb) ' - EEG Time Series']);
    xlabel('Time (seconds)');
    ylabel('Channel (stacked)');
    set(gca, 'YTick', 10*(1:64), 'YTickLabel', 1:64);
    grid on;

    % 保存 PNG 图片
    saveas(gcf, fullfile(output_folder, ['Subject_' num2str(sb) '.png']));
    close(gcf);
end

disp('EEG 时序图已保存到 ../EEG_TimeSeries_Plots/');
