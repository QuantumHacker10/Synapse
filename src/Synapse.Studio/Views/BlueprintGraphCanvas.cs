using System;
using System.Collections.Generic;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using Synapse.Runtime;

namespace Synapse.Studio.Views
{
    /// <summary>Minimal visual blueprint graph editor (nodes + exec edges).</summary>
    public sealed class BlueprintGraphCanvas : Control
    {
        public static readonly StyledProperty<BlueprintDocument?> DocumentProperty =
            AvaloniaProperty.Register<BlueprintGraphCanvas, BlueprintDocument?>(nameof(Document));

        public static readonly StyledProperty<Guid?> SelectedNodeIdProperty =
            AvaloniaProperty.Register<BlueprintGraphCanvas, Guid?>(nameof(SelectedNodeId));

        private Point _pan;
        private bool _draggingNode;
        private Guid? _dragNodeId;
        private Point _dragStart;
        private Guid? _connectFromId;

        public BlueprintDocument? Document
        {
            get => GetValue(DocumentProperty);
            set => SetValue(DocumentProperty, value);
        }

        public Guid? SelectedNodeId
        {
            get => GetValue(SelectedNodeIdProperty);
            set => SetValue(SelectedNodeIdProperty, value);
        }

        public event EventHandler? DocumentChanged;

        public BlueprintGraphCanvas()
        {
            ClipToBounds = true;
            Focusable = true;
        }

        protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
        {
            base.OnPropertyChanged(change);
            if (change.Property == DocumentProperty)
                InvalidateVisual();
        }

        public override void Render(DrawingContext context)
        {
            base.Render(context);
            var doc = Document;
            if (doc == null) return;

            context.FillRectangle(new SolidColorBrush(Color.Parse("#0E1218")), Bounds);

            var nodeMap = new Dictionary<Guid, BlueprintNode>();
            foreach (var node in doc.Nodes)
                nodeMap[node.Id] = node;

            foreach (var edge in doc.Edges)
            {
                if (!nodeMap.TryGetValue(edge.FromNodeId, out var from) ||
                    !nodeMap.TryGetValue(edge.ToNodeId, out var to))
                    continue;

                var p0 = new Point(from.X + 150 + _pan.X, from.Y + 28 + _pan.Y);
                var p1 = new Point(to.X + _pan.X, to.Y + 28 + _pan.Y);
                var pen = new Pen(new SolidColorBrush(Color.Parse("#45E0B8")), 2);
                context.DrawLine(pen, p0, p1);
            }

            foreach (var node in doc.Nodes)
            {
                var rect = new Rect(node.X + _pan.X, node.Y + _pan.Y, 150, 56);
                var fill = node.Id == SelectedNodeId
                    ? new SolidColorBrush(Color.Parse("#243044"))
                    : new SolidColorBrush(Color.Parse("#161B22"));
                context.FillRectangle(fill, rect, 6);
                var border = node.Id == SelectedNodeId
                    ? new SolidColorBrush(Color.Parse("#45E0B8"))
                    : new SolidColorBrush(Color.Parse("#2A3444"));
                context.DrawRectangle(new Pen(border, 1.5), rect, 6);
                context.DrawText(new FormattedText(
                    $"{node.Kind}\n{node.Title}",
                    System.Globalization.CultureInfo.CurrentCulture,
                    FlowDirection.LeftToRight,
                    new Typeface("Segoe UI"),
                    11,
                    new SolidColorBrush(Color.Parse("#E8EEF7"))), new Point(rect.X + 8, rect.Y + 8));
            }
        }

        protected override void OnPointerPressed(PointerPressedEventArgs e)
        {
            base.OnPointerPressed(e);
            Focus();
            var doc = Document;
            if (doc == null) return;
            var pos = e.GetPosition(this);

            foreach (var node in doc.Nodes)
            {
                var rect = new Rect(node.X + _pan.X, node.Y + _pan.Y, 150, 56);
                if (!rect.Contains(pos)) continue;

                SelectedNodeId = node.Id;

                if (e.KeyModifiers.HasFlag(KeyModifiers.Control))
                {
                    if (_connectFromId == null) _connectFromId = node.Id;
                    else if (_connectFromId != node.Id)
                    {
                        doc.Edges.Add(new BlueprintEdge
                        {
                            FromNodeId = _connectFromId.Value,
                            FromPin = 0,
                            ToNodeId = node.Id,
                            ToPin = 0
                        });
                        _connectFromId = null;
                        DocumentChanged?.Invoke(this, EventArgs.Empty);
                    }
                }
                else
                {
                    _draggingNode = true;
                    _dragNodeId = node.Id;
                    _dragStart = pos;
                }

                InvalidateVisual();
                e.Handled = true;
                return;
            }
        }

        protected override void OnPointerMoved(PointerEventArgs e)
        {
            base.OnPointerMoved(e);
            var doc = Document;
            if (doc == null || !_draggingNode || _dragNodeId == null) return;

            var pos = e.GetPosition(this);
            var delta = pos - _dragStart;
            _dragStart = pos;
            var node = doc.Nodes.Find(n => n.Id == _dragNodeId);
            if (node == null) return;
            node.X += (float)delta.X;
            node.Y += (float)delta.Y;
            InvalidateVisual();
            DocumentChanged?.Invoke(this, EventArgs.Empty);
        }

        protected override void OnPointerReleased(PointerReleasedEventArgs e)
        {
            base.OnPointerReleased(e);
            _draggingNode = false;
            _dragNodeId = null;
        }

        protected override void OnPointerWheelChanged(PointerWheelEventArgs e)
        {
            base.OnPointerWheelChanged(e);
            _pan += new Vector(0, e.Delta.Y * 20);
            InvalidateVisual();
        }
    }
}
