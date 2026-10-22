{
 "patcher": {
  "fileversion": 1,
  "appversion": {
   "major": 9,
   "minor": 2,
   "revision": 0,
   "architecture": "x64",
   "modernui": 1
  },
  "classnamespace": "box",
  "rect": [
   40.0,
   80.0,
   1220.0,
   760.0
  ],
  "gridsize": [
   15.0,
   15.0
  ],
  "boxes": [
   {
    "box": {
     "maxclass": "comment",
     "text": "Performance RNN in real time: the model plays, you steer it",
     "patching_rect": [
      15,
      12,
      900,
      22.0
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "fontsize": 16.0,
     "fontface": 1,
     "id": "obj-1"
    }
   },
   {
    "box": {
     "maxclass": "panel",
     "patching_rect": [
      15,
      50,
      360,
      300
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "background": 1,
     "bgcolor": [
      0.93,
      0.93,
      0.93,
      1.0
     ],
     "border": 0,
     "rounded": 8,
     "mode": 0,
     "id": "obj-2"
    }
   },
   {
    "box": {
     "maxclass": "comment",
     "text": "1 \u00b7 SERVER",
     "patching_rect": [
      25,
      57,
      330,
      22.0
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "fontsize": 14.0,
     "fontface": 1,
     "id": "obj-3"
    }
   },
   {
    "box": {
     "maxclass": "comment",
     "text": "Starts prnn serve with the spawn object, from the folder of this patch. Its messages appear below and in the Max console.",
     "patching_rect": [
      25,
      79,
      330,
      48
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "id": "obj-4"
    }
   },
   {
    "box": {
     "maxclass": "message",
     "text": "../../.venv/bin/prnn serve -s ../../models/bach.sess",
     "patching_rect": [
      25,
      140,
      330,
      22.0
     ],
     "numinlets": 2,
     "numoutlets": 1,
     "outlettype": [
      ""
     ],
     "id": "obj-5"
    }
   },
   {
    "box": {
     "maxclass": "message",
     "text": "stop",
     "patching_rect": [
      25,
      175,
      36,
      22.0
     ],
     "numinlets": 2,
     "numoutlets": 1,
     "outlettype": [
      ""
     ],
     "id": "obj-6"
    }
   },
   {
    "box": {
     "maxclass": "newobj",
     "text": "spawn",
     "patching_rect": [
      25,
      210,
      60,
      22.0
     ],
     "numinlets": 1,
     "numoutlets": 2,
     "outlettype": [
      "",
      ""
     ],
     "id": "obj-7"
    }
   },
   {
    "box": {
     "maxclass": "newobj",
     "text": "print prnn",
     "patching_rect": [
      25,
      250,
      70,
      22.0
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "id": "obj-8"
    }
   },
   {
    "box": {
     "maxclass": "toggle",
     "parameter_enable": 0,
     "patching_rect": [
      145,
      250,
      24.0,
      24.0
     ],
     "numinlets": 1,
     "numoutlets": 1,
     "outlettype": [
      "int"
     ],
     "id": "obj-9"
    }
   },
   {
    "box": {
     "maxclass": "comment",
     "text": "running",
     "patching_rect": [
      175,
      252,
      70,
      20.0
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "id": "obj-10"
    }
   },
   {
    "box": {
     "maxclass": "comment",
     "text": "The model is models/bach.sess in the repository; change the message to use another one.",
     "patching_rect": [
      25,
      290,
      330,
      33
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "id": "obj-11"
    }
   },
   {
    "box": {
     "maxclass": "panel",
     "patching_rect": [
      390,
      50,
      420,
      700
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "background": 1,
     "bgcolor": [
      0.93,
      0.93,
      0.93,
      1.0
     ],
     "border": 0,
     "rounded": 8,
     "mode": 0,
     "id": "obj-12"
    }
   },
   {
    "box": {
     "maxclass": "comment",
     "text": "2 \u00b7 CONTROLS",
     "patching_rect": [
      400,
      57,
      390,
      22.0
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "fontsize": 14.0,
     "fontface": 1,
     "id": "obj-13"
    }
   },
   {
    "box": {
     "maxclass": "comment",
     "text": "Every control goes to the server (port 9010) through s prnn.osc. Changes are heard after about half a second.",
     "patching_rect": [
      400,
      79,
      390,
      48
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "id": "obj-14"
    }
   },
   {
    "box": {
     "maxclass": "toggle",
     "parameter_enable": 0,
     "patching_rect": [
      400,
      135,
      30,
      30
     ],
     "numinlets": 1,
     "numoutlets": 1,
     "outlettype": [
      "int"
     ],
     "id": "obj-15"
    }
   },
   {
    "box": {
     "maxclass": "comment",
     "text": "play",
     "patching_rect": [
      438,
      140,
      50,
      20.0
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "id": "obj-16"
    }
   },
   {
    "box": {
     "maxclass": "newobj",
     "text": "prepend /play",
     "patching_rect": [
      400,
      175,
      87,
      22.0
     ],
     "numinlets": 1,
     "numoutlets": 1,
     "outlettype": [
      ""
     ],
     "id": "obj-17"
    }
   },
   {
    "box": {
     "maxclass": "newobj",
     "text": "s prnn.osc",
     "patching_rect": [
      400,
      205,
      70,
      22.0
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "id": "obj-18"
    }
   },
   {
    "box": {
     "maxclass": "message",
     "text": "/reset",
     "patching_rect": [
      620,
      135,
      48,
      22.0
     ],
     "numinlets": 2,
     "numoutlets": 1,
     "outlettype": [
      ""
     ],
     "id": "obj-19"
    }
   },
   {
    "box": {
     "maxclass": "comment",
     "text": "start again",
     "patching_rect": [
      675,
      135,
      90,
      20.0
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "id": "obj-20"
    }
   },
   {
    "box": {
     "maxclass": "newobj",
     "text": "s prnn.osc",
     "patching_rect": [
      620,
      175,
      70,
      22.0
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "id": "obj-21"
    }
   },
   {
    "box": {
     "maxclass": "newobj",
     "text": "loadmess 1.",
     "patching_rect": [
      400,
      250,
      76,
      22.0
     ],
     "numinlets": 1,
     "numoutlets": 1,
     "outlettype": [
      ""
     ],
     "id": "obj-22"
    }
   },
   {
    "box": {
     "maxclass": "flonum",
     "parameter_enable": 0,
     "patching_rect": [
      400,
      280,
      55.0,
      22.0
     ],
     "numinlets": 1,
     "numoutlets": 2,
     "outlettype": [
      "",
      "bang"
     ],
     "format": 6,
     "id": "obj-23"
    }
   },
   {
    "box": {
     "maxclass": "comment",
     "text": "temperature: below 1 more predictable, above 1 more surprising",
     "patching_rect": [
      465,
      275,
      300,
      33
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "id": "obj-24"
    }
   },
   {
    "box": {
     "maxclass": "newobj",
     "text": "prepend /temperature",
     "patching_rect": [
      400,
      310,
      126,
      22.0
     ],
     "numinlets": 1,
     "numoutlets": 1,
     "outlettype": [
      ""
     ],
     "id": "obj-25"
    }
   },
   {
    "box": {
     "maxclass": "newobj",
     "text": "s prnn.osc",
     "patching_rect": [
      570,
      338,
      70,
      22.0
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "id": "obj-26"
    }
   },
   {
    "box": {
     "maxclass": "newobj",
     "text": "loadmess 6.",
     "patching_rect": [
      400,
      355,
      76,
      22.0
     ],
     "numinlets": 1,
     "numoutlets": 1,
     "outlettype": [
      ""
     ],
     "id": "obj-27"
    }
   },
   {
    "box": {
     "maxclass": "flonum",
     "parameter_enable": 0,
     "patching_rect": [
      400,
      385,
      55.0,
      22.0
     ],
     "numinlets": 1,
     "numoutlets": 2,
     "outlettype": [
      "",
      "bang"
     ],
     "format": 6,
     "id": "obj-28"
    }
   },
   {
    "box": {
     "maxclass": "comment",
     "text": "notes per second (density)",
     "patching_rect": [
      465,
      385,
      220,
      20.0
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "id": "obj-29"
    }
   },
   {
    "box": {
     "maxclass": "newobj",
     "text": "prepend /density",
     "patching_rect": [
      400,
      415,
      104,
      22.0
     ],
     "numinlets": 1,
     "numoutlets": 1,
     "outlettype": [
      ""
     ],
     "id": "obj-30"
    }
   },
   {
    "box": {
     "maxclass": "newobj",
     "text": "s prnn.osc",
     "patching_rect": [
      570,
      443,
      70,
      22.0
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "id": "obj-31"
    }
   },
   {
    "box": {
     "maxclass": "newobj",
     "text": "loadbang",
     "patching_rect": [
      400,
      460,
      59,
      22.0
     ],
     "numinlets": 1,
     "numoutlets": 1,
     "outlettype": [
      "bang"
     ],
     "id": "obj-32"
    }
   },
   {
    "box": {
     "maxclass": "message",
     "text": "1 1 1 1 1 1 1 1 1 1 1 1",
     "patching_rect": [
      400,
      490,
      160,
      22.0
     ],
     "numinlets": 2,
     "numoutlets": 1,
     "outlettype": [
      ""
     ],
     "id": "obj-33"
    }
   },
   {
    "box": {
     "maxclass": "multislider",
     "parameter_enable": 0,
     "patching_rect": [
      400,
      520,
      260,
      60
     ],
     "numinlets": 1,
     "numoutlets": 2,
     "outlettype": [
      "",
      ""
     ],
     "size": 12,
     "setminmax": [
      0.0,
      1.0
     ],
     "setstyle": 1,
     "slidercolor": [
      0.35,
      0.45,
      0.07,
      1.0
     ],
     "id": "obj-34"
    }
   },
   {
    "box": {
     "maxclass": "comment",
     "text": "pitch classes C C# D D# E F F# G G# A A# B",
     "patching_rect": [
      400,
      582,
      270,
      20.0
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "id": "obj-35"
    }
   },
   {
    "box": {
     "maxclass": "newobj",
     "text": "prepend /histogram",
     "patching_rect": [
      400,
      610,
      115,
      22.0
     ],
     "numinlets": 1,
     "numoutlets": 1,
     "outlettype": [
      ""
     ],
     "id": "obj-36"
    }
   },
   {
    "box": {
     "maxclass": "newobj",
     "text": "s prnn.osc",
     "patching_rect": [
      570,
      638,
      70,
      22.0
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "id": "obj-37"
    }
   },
   {
    "box": {
     "maxclass": "newobj",
     "text": "loadmess 1",
     "patching_rect": [
      690,
      490,
      70,
      22.0
     ],
     "numinlets": 1,
     "numoutlets": 1,
     "outlettype": [
      ""
     ],
     "id": "obj-38"
    }
   },
   {
    "box": {
     "maxclass": "toggle",
     "parameter_enable": 0,
     "patching_rect": [
      690,
      520,
      24.0,
      24.0
     ],
     "numinlets": 1,
     "numoutlets": 1,
     "outlettype": [
      "int"
     ],
     "id": "obj-39"
    }
   },
   {
    "box": {
     "maxclass": "comment",
     "text": "use the controls",
     "patching_rect": [
      718,
      522,
      90,
      33
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "id": "obj-40"
    }
   },
   {
    "box": {
     "maxclass": "newobj",
     "text": "prepend /controls",
     "patching_rect": [
      690,
      560,
      109,
      22.0
     ],
     "numinlets": 1,
     "numoutlets": 1,
     "outlettype": [
      ""
     ],
     "id": "obj-41"
    }
   },
   {
    "box": {
     "maxclass": "newobj",
     "text": "s prnn.osc",
     "patching_rect": [
      690,
      590,
      70,
      22.0
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "id": "obj-42"
    }
   },
   {
    "box": {
     "maxclass": "newobj",
     "text": "r prnn.osc",
     "patching_rect": [
      400,
      675,
      70,
      22.0
     ],
     "numinlets": 0,
     "numoutlets": 1,
     "outlettype": [
      ""
     ],
     "id": "obj-43"
    }
   },
   {
    "box": {
     "maxclass": "newobj",
     "text": "udpsend 127.0.0.1 9010",
     "patching_rect": [
      400,
      705,
      150,
      22.0
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "id": "obj-44"
    }
   },
   {
    "box": {
     "maxclass": "panel",
     "patching_rect": [
      825,
      50,
      360,
      330
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "background": 1,
     "bgcolor": [
      0.93,
      0.93,
      0.93,
      1.0
     ],
     "border": 0,
     "rounded": 8,
     "mode": 0,
     "id": "obj-45"
    }
   },
   {
    "box": {
     "maxclass": "comment",
     "text": "3 \u00b7 NOTES",
     "patching_rect": [
      835,
      57,
      330,
      22.0
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "fontsize": 14.0,
     "fontface": 1,
     "id": "obj-46"
    }
   },
   {
    "box": {
     "maxclass": "comment",
     "text": "Received on port 9011 and played by noteout (double-click it to choose a synthesizer).",
     "patching_rect": [
      835,
      79,
      330,
      33
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "id": "obj-47"
    }
   },
   {
    "box": {
     "maxclass": "newobj",
     "text": "udpreceive 9011",
     "patching_rect": [
      835,
      130,
      98,
      22.0
     ],
     "numinlets": 1,
     "numoutlets": 1,
     "outlettype": [
      ""
     ],
     "id": "obj-48"
    }
   },
   {
    "box": {
     "maxclass": "newobj",
     "text": "route /note_on /note_off /playing",
     "patching_rect": [
      835,
      165,
      230,
      22.0
     ],
     "numinlets": 1,
     "numoutlets": 4,
     "outlettype": [
      "",
      "",
      "",
      ""
     ],
     "id": "obj-49"
    }
   },
   {
    "box": {
     "maxclass": "message",
     "text": "$1 0",
     "patching_rect": [
      920,
      200,
      36,
      22.0
     ],
     "numinlets": 2,
     "numoutlets": 1,
     "outlettype": [
      ""
     ],
     "id": "obj-50"
    }
   },
   {
    "box": {
     "maxclass": "kslider",
     "parameter_enable": 0,
     "patching_rect": [
      835,
      240,
      330,
      50
     ],
     "numinlets": 2,
     "numoutlets": 2,
     "outlettype": [
      "int",
      "int"
     ],
     "mode": 1,
     "id": "obj-51"
    }
   },
   {
    "box": {
     "maxclass": "newobj",
     "text": "noteout",
     "patching_rect": [
      835,
      315,
      53,
      22.0
     ],
     "numinlets": 3,
     "numoutlets": 0,
     "id": "obj-52"
    }
   },
   {
    "box": {
     "maxclass": "toggle",
     "parameter_enable": 0,
     "patching_rect": [
      1075,
      200,
      24.0,
      24.0
     ],
     "numinlets": 1,
     "numoutlets": 1,
     "outlettype": [
      "int"
     ],
     "id": "obj-53"
    }
   },
   {
    "box": {
     "maxclass": "comment",
     "text": "playing",
     "patching_rect": [
      1103,
      202,
      60,
      20.0
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "id": "obj-54"
    }
   },
   {
    "box": {
     "maxclass": "panel",
     "patching_rect": [
      15,
      365,
      360,
      225
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "background": 1,
     "bgcolor": [
      0.88,
      0.91,
      0.95,
      1.0
     ],
     "border": 0,
     "rounded": 8,
     "mode": 0,
     "id": "obj-55"
    }
   },
   {
    "box": {
     "maxclass": "comment",
     "text": "How to use",
     "patching_rect": [
      25,
      372,
      330,
      22.0
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "fontsize": 14.0,
     "fontface": 1,
     "id": "obj-56"
    }
   },
   {
    "box": {
     "maxclass": "comment",
     "text": "1. Once: install the spawn package and train or download a model (README).\n2. Click the message in section 1: the console shows 'listening on port 9010'.\n3. Turn on play. Move temperature, density and the pitch classes.\n4. stop (or closing the patch) ends the server and releases every note.",
     "patching_rect": [
      25,
      397,
      335,
      160
     ],
     "numinlets": 1,
     "numoutlets": 0,
     "id": "obj-57"
    }
   }
  ],
  "lines": [
   {
    "patchline": {
     "source": [
      "obj-5",
      0
     ],
     "destination": [
      "obj-7",
      0
     ]
    }
   },
   {
    "patchline": {
     "source": [
      "obj-6",
      0
     ],
     "destination": [
      "obj-7",
      0
     ]
    }
   },
   {
    "patchline": {
     "source": [
      "obj-7",
      0
     ],
     "destination": [
      "obj-8",
      0
     ]
    }
   },
   {
    "patchline": {
     "source": [
      "obj-7",
      1
     ],
     "destination": [
      "obj-9",
      0
     ]
    }
   },
   {
    "patchline": {
     "source": [
      "obj-15",
      0
     ],
     "destination": [
      "obj-17",
      0
     ]
    }
   },
   {
    "patchline": {
     "source": [
      "obj-17",
      0
     ],
     "destination": [
      "obj-18",
      0
     ]
    }
   },
   {
    "patchline": {
     "source": [
      "obj-19",
      0
     ],
     "destination": [
      "obj-21",
      0
     ]
    }
   },
   {
    "patchline": {
     "source": [
      "obj-25",
      0
     ],
     "destination": [
      "obj-26",
      0
     ]
    }
   },
   {
    "patchline": {
     "source": [
      "obj-22",
      0
     ],
     "destination": [
      "obj-23",
      0
     ]
    }
   },
   {
    "patchline": {
     "source": [
      "obj-23",
      0
     ],
     "destination": [
      "obj-25",
      0
     ]
    }
   },
   {
    "patchline": {
     "source": [
      "obj-30",
      0
     ],
     "destination": [
      "obj-31",
      0
     ]
    }
   },
   {
    "patchline": {
     "source": [
      "obj-27",
      0
     ],
     "destination": [
      "obj-28",
      0
     ]
    }
   },
   {
    "patchline": {
     "source": [
      "obj-28",
      0
     ],
     "destination": [
      "obj-30",
      0
     ]
    }
   },
   {
    "patchline": {
     "source": [
      "obj-36",
      0
     ],
     "destination": [
      "obj-37",
      0
     ]
    }
   },
   {
    "patchline": {
     "source": [
      "obj-32",
      0
     ],
     "destination": [
      "obj-33",
      0
     ]
    }
   },
   {
    "patchline": {
     "source": [
      "obj-33",
      0
     ],
     "destination": [
      "obj-34",
      0
     ]
    }
   },
   {
    "patchline": {
     "source": [
      "obj-34",
      0
     ],
     "destination": [
      "obj-36",
      0
     ]
    }
   },
   {
    "patchline": {
     "source": [
      "obj-41",
      0
     ],
     "destination": [
      "obj-42",
      0
     ]
    }
   },
   {
    "patchline": {
     "source": [
      "obj-38",
      0
     ],
     "destination": [
      "obj-39",
      0
     ]
    }
   },
   {
    "patchline": {
     "source": [
      "obj-39",
      0
     ],
     "destination": [
      "obj-41",
      0
     ]
    }
   },
   {
    "patchline": {
     "source": [
      "obj-43",
      0
     ],
     "destination": [
      "obj-44",
      0
     ]
    }
   },
   {
    "patchline": {
     "source": [
      "obj-48",
      0
     ],
     "destination": [
      "obj-49",
      0
     ]
    }
   },
   {
    "patchline": {
     "source": [
      "obj-49",
      0
     ],
     "destination": [
      "obj-51",
      0
     ]
    }
   },
   {
    "patchline": {
     "source": [
      "obj-49",
      0
     ],
     "destination": [
      "obj-52",
      0
     ]
    }
   },
   {
    "patchline": {
     "source": [
      "obj-49",
      1
     ],
     "destination": [
      "obj-50",
      0
     ]
    }
   },
   {
    "patchline": {
     "source": [
      "obj-50",
      0
     ],
     "destination": [
      "obj-51",
      0
     ]
    }
   },
   {
    "patchline": {
     "source": [
      "obj-50",
      0
     ],
     "destination": [
      "obj-52",
      0
     ]
    }
   },
   {
    "patchline": {
     "source": [
      "obj-49",
      2
     ],
     "destination": [
      "obj-53",
      0
     ]
    }
   }
  ]
 }
}