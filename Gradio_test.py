import random



# ------------------sample from gradio ------------------
import math

import pandas as pd

import gradio as gr
import datetime
import numpy as np


def get_time():
    now_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S  %A")
    return now_time


plot_end = 2 * math.pi


def get_plot(period=1):
    global plot_end
    x = np.arange(plot_end - 2 * math.pi, plot_end, 0.02)
    y = np.sin(2 * math.pi * period * x)
    # 非常非常奇葩的版本兼容的问题: gradio的文档中,并没有定义gr.LinePlot.update方法来更新LinePlot的参数;
    # 而本代码中,LinePlot先行运行一次,然后每次对诸参数进行更新, 此处需要gr.LinePlot.update(),里面放置需要更新的形参;
    # 而gradio文档中, 并不需要update(),而仅仅是gr.LinePlot()里面放置更新的参数,重新运行一遍.
    # 以上问题花费了1天半,找出问题解决
    update = gr.LinePlot.update(
        value=pd.DataFrame({"X": x, "Y": y}),
        x="X",
        y="Y",
        title="Plot (updates every second)",
        width=600,
        height=350,
    )
    plot_end += 2 * math.pi
    if plot_end > 1000:
        plot_end = 2 * math.pi
    return update


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            c_time2 = gr.Textbox(label="Current Time refreshed every second")
            gr.Textbox(
                "Change the value of the slider to automatically update the plot",
                label="",
            )
            period = gr.Slider(
                label="Period of plot", value=1, minimum=0, maximum=10, step=1
            )
            plot = gr.LinePlot(show_label=False)
        with gr.Column():
            name = gr.Textbox(label="Enter your name")
            greeting = gr.Textbox(label="Greeting")
            button = gr.Button(value="Greet")
            button.click(lambda s: f"Hello {s}", name, greeting)

    demo.load(get_time, None, c_time2, every=1)
    dep = demo.load(get_plot, None, plot, every=1, )
    period.change(get_plot, period, plot, every=1, cancels=[dep])
#
# if __name__ == "__main__":
#     demo.queue().launch()


# 以下为另一个例子,可以忽略


## Or generate your own fake data, here's an example for stocks:
#
stocks = pd.DataFrame(
    {
        "symbol": [
            random.choice(
                [
                    "MSFT",
                    "AAPL",
                    "AMZN",
                    "IBM",
                    "GOOG",
                ]
            )
            for _ in range(120)
        ],
        "date": [
            pd.Timestamp(year=2000 + i, month=j, day=1)
            for i in range(10)
            for j in range(1, 13)
        ],
        "price": [random.randint(10, 200) for _ in range(120)],
    }
)


def line_plot_fn(dataset):
    if dataset == "stocks":
        # 非常非常奇葩的版本兼容的问题: gradio的文档中,并没有定义gr.LinePlot.update方法来更新LinePlot的参数;
        # 而本代码中,LinePlot先行运行一次,然后每次对诸参数进行更新, 此处需要gr.LinePlot.update(),里面放置需要更新的形参;
        # 而gradio文档中, 并不需要update(),而仅仅是gr.LinePlot()里面放置更新的参数,重新运行一遍.
        # 以上问题花费了1天半,找出问题解决
        return gr.LinePlot.update(
            value=stocks,
            x="date",
            y="price",
            color="symbol",
            color_legend_position="bottom",
            title="Stock Prices",
            tooltip=["date", "price", "symbol"],
            height=300,
            width=500,
        )
        # return stocks
    elif dataset == "symbol":
        # 非常非常奇葩的版本兼容的问题: gradio的文档中,并没有定义gr.LinePlot.update方法来更新LinePlot的参数;
        # 而本代码中,LinePlot先行运行一次,然后每次对诸参数进行更新, 此处需要gr.LinePlot.update(),里面放置需要更新的形参;
        # 而gradio文档中, 并不需要update(),而仅仅是gr.LinePlot()里面放置更新的参数,重新运行一遍.
        # 以上问题花费了1天半,找出问题解决
        return gr.update(
            value=stocks,
            x="date",
            y="price",
            color="symbol",
            color_legend_position="bottom",
            title="Stock Symbol",
            tooltip=["date", "price", "symbol"],
            height=300,
            width=500,
        )


with gr.Blocks() as line_plot:
    with gr.Row():
        with gr.Column():
            dataset = gr.Dropdown(
                choices=["stocks", "symbol"],
                value="stocks",
            )
        with gr.Column():
            plot = gr.LinePlot(interactive=True)

    line_plot.load(fn=line_plot_fn, inputs=dataset, outputs=plot)
    dataset.change(line_plot_fn, inputs=dataset, outputs=plot)

if __name__ == "__main__":
    demo.queue().launch(debug=True)
