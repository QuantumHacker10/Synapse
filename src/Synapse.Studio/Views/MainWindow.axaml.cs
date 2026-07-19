using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Synapse.Studio.ViewModels;

namespace Synapse.Studio.Views
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            Loaded += OnLoaded;
        }

        private void OnLoaded(object? sender, RoutedEventArgs e)
        {
            if (DataContext is MainWindowViewModel vm && BlueprintCanvas != null)
                BlueprintCanvas.Document = vm.Blueprint;
        }

        private void OnBlueprintDocumentChanged(object? sender, System.EventArgs e)
        {
            if (DataContext is MainWindowViewModel vm)
                vm.NotifyBlueprintChanged();
        }

        private void OnWindowKeyDown(object? sender, KeyEventArgs e)
        {
            if (e.Key != Key.Space || DataContext is not MainWindowViewModel vm)
                return;
            if (e.Source is TextBox)
                return;

            vm.TogglePlayCommand.Execute(null);
            e.Handled = true;
        }
    }
}
