#![warn(clippy::all)]

use grafix_toolbox::{asyn::*, gui::*, lib::*, math::*, *};
use GL::{atlas::*, font::*, mesh::*, pbrt::*, window::*, *};

fn main() {
	LOGGER!(logging::Term, INFO);


	let mut window = TIMER!(window, {
		let win = EXPECT!(Window::get((50, 50, 1600, 900), "Engine"));
		GLEnable!(DEPTH_TEST, TEXTURE_CUBE_MAP_SEAMLESS);
		GL::BlendFunc::Set((gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA));
		GL::DepthFunc::Set(gl::LESS);
		win
	});

	ShaderManager::Initialize(&mut window);
	ShaderManager::LoadSources("shd_pbrt.glsl");

	let font = {
		let alphabet = (9..=9).chain(32_u8..127).map(|n| n as char).collect::<String>() + "ёйцукенгшщзхъфывапролджэячсмитьбюЁЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ";
		Font::new_cached("UbuntuMono-R", alphabet)
	};
	let large_text = "Functional textbox is capable of showing millions of lines with negligeble cpu usage!(so long as we're not editing, but that can be solved functionally as well)\nBiggest issue is collecting the text to display, really - you can set large text range to 1000000 in release.\n\nUpper bar allows to drag layout around, pip on the bottom right resizes it.\n\n\n".repeat(100);

	let popups_text = "With my approach, you can implement context popups in your ui, in under 250 lines of code. I mean, something like this takes a very complex toolkit and A LOT of code, traditionally. This is the power of functional architecture. The implementation also has ample space for memoizations, in case *fast* isn't fast enough.\n\n\n".repeat(2);

	let db = [
		(
			"context popups",
			"Pioneered in crusader kings 3, afaik, this is a really cool ui element.\n\nBTW these context popups are recursive!",
		),
		("ui", "This very thing"),
		(
			"my approach",
			"Look into toolbox/gui/elements.\nYes there're some safety overrides there,\nbut this is my project and i can do whatever i want in it!",
		),
		(
			"functional architecture",
			"I've noticed that people often make a fantasy OOP architecture, maybe even write a UML diag, and then forget about it. Functional architecture means writing out the actual pipeline of the data transformations your client wants, and then implementing it recursively with simplest possible functions. When you have MVP, you can optimise your functions with memoization, and then maybe even with SIMD or a gpu. Optimisation on a directed graph is very simple!",
		),
	];
	let db = &HyperDB::new(&font, 0.05, db);

	let mut renderer = Renderer::new(Theme {
		easing: 10.,
		bg: (0.2, 0.2, 0.2, 0.7),
		bg_focus: hex_to_rgba(0x596475A0),
		fg: hex_to_rgba(0x626975FF),
		fg_focus: hex_to_rgba(0x461E5CCF),
		highlight: (0.9, 0.4, 0.1, 1.),
		text: (1., 0.9, 0.9, 0.9),
		text_focus: (1., 1., 1., 1.),
		text_highlight: (0.2, 0.2, 0.2, 1.),
		font,
		font_size: 0.8,
	});

	let atlas = TexAtlas::<RGBA>::new();
	let mut spinner = EXPECT!(Animation::from_file("spinner", &atlas));

	let (skybox, brdf_lut) = (EnvTex::from(EXPECT!(Environment::new_cached("lythwood_lounge_4k"))), Environment::lut_cached());

	let (mut skybox_shd, mut render_shd) = (
		EXPECT!(Shader::new(("vs_skybox", "ps_skybox"))),
		EXPECT!(Shader::new(("vs_material_based_render", "ps_material_based_render"))),
	);

	let (sampl, mipmapped) = (
		&Sampler::linear(),
		&Sampler!(
			(TEXTURE_WRAP_R, CLAMP_TO_EDGE),
			(TEXTURE_WRAP_S, CLAMP_TO_EDGE),
			(TEXTURE_WRAP_T, CLAMP_TO_EDGE),
			(TEXTURE_MIN_FILTER, LINEAR_MIPMAP_LINEAR)
		),
	);


	Layout::storage(ID!("menu")).pos = (-10., -10.);
	Layout::storage(ID!("menu")).size = (1.5, 2.);
	HyperText::storage(ID!("hypertext")).text = popups_text.into();
	TextEdit::storage(ID!("text")).text = large_text.into();
	Slider::storage(ID!("metallicity")).pip_pos = 0.885;
	Slider::storage(ID!("roughness")).pip_pos = 0.286;
	Slider::storage(ID!("rotation")).pip_pos = 0.1;

	let show_tooltip = |hovered, p: Vec2, msg: &str, r: &mut RenderLock| {
		if timeout(hovered) {
			let size = (0.03, 0.1).mul((msg.len(), 1));
			let _c = r.clip(p, size);
			r.Label(ID!("Tooltip"), p.sum(size.mul(0.05)), size, msg);
		}
	};

	let mut exit = false;
	let mut mouse_pos = (0., 0.);
	let mut options = ["stanford_dragon", "dr", "tyra", "not a model"].map(String::from);
	let mut demo_mesh = DeferredMesh::default();
	let mut loading_in_progress;
	let mut magnification = 0.3;
	let mut rotation = 0_f32;

	let mut lights = UniformArr::new((vec![0.; 4 + 4 * 20 * 2], gl::DYNAMIC_STORAGE_BIT | gl::MAP_WRITE_BIT));

	for i in (0..100).chain((0..100).step_by(1).rev()).map(|i| f32(i) / 100.).cycle() {
		let mut r = renderer.lock();

		let mut cam1 = Camera::zero((&window, (70, 10)));
		{
			window.bind();
			window.clear((0, 1));

			let metallicity = 0.995 * r.Slider(ID!("metallicity"), (0.3, -0.88), (1., 0.05), 0.05);
			let roughness = (0.05 + r.Slider(ID!("roughness"), (0.3, -0.94), (1., 0.05), 0.05)) / 1.05;
			r.Label(ID!("metal_v"), (1.31, -0.88), (0.3, 0.05), &format!("Metal: {:.3}", metallicity));
			r.Label(ID!("rough_v"), (1.31, -0.94), (0.3, 0.05), &format!("Rough: {:.3}", roughness));
			let model_name = Selector::storage(ID!("model")).choice;
			let model_name = options[model_name].clone();
			r.Selector(ID!("model"), (1.31, -0.82), (0.6, 0.1), &mut options);

			let t = 1. - i;

			let (v3, v4) = (la::Vec3::new, la::Vec4::new);
			use la::Mat4 as m4;
			let m = magnification;
			let model = &la::translate(la::rotate(la::scale(la::identity(), v3(m, m, m)), 90_f32.to_radians(), v3(-1., 0., 0.)), v3(0., 0., -0.3));
			let (c, s, cam_world) = {
				rotation += r.Slider(ID!("rotation"), (0.3, -1.), (1., 0.05), 0.05) * 0.01;
				let (c, s) = (rotation.cos(), rotation.sin()).mul(0.3);
				let t = t.to_radians() * 90.;
				(t.cos(), t.sin(), (s, 0., c))
			};

			cam1.set_view((cam_world, Vec3::zero(), (0, 1, 0)));
			let view = m4::to(cam1.V());

			{
				let l = lights.array.MapRangeMut((0, 4 + 4 * 4 * 2)).mem();
				l[0] = f32::from_bits(4);

				let lights = [
					Vec4(view * v4(6. * c, 6. * s, 0., 1.)),
					(1., 0., 0., 2.),
					Vec4(view * v4(2. * c, 0., 2. * s, 1.)),
					(0., 0., 1., 2.),
					Vec4(view * v4(c, s, -2., 1.)),
					(1., 0., 1., 2.),
					Vec4(view * v4(-1.5 * s, 1. * c, -2., 1.)),
					(0., 1., 0., 2.),
				]
				.flatten();

				l[4..].copy_from_slice(&lights);
			}

			let (irr, spec, lut, lights) = (skybox.irradiance.Bind(sampl), skybox.specular.Bind(mipmapped), brdf_lut.Bind(sampl), lights.Bind());
			let _ = Uniforms!(
				render_shd,
				("irradiance_cubetex", &irr),
				("specular_cubetex", &spec),
				("brdf_lut_tex", &lut),
				("MVPMat", cam1.MVP(model)),
				("ModelViewMat", cam1.MV(model)),
				("NormalViewMat", cam1.NV(model)),
				("NormalMat", cam1.N(model)),
				("iCameraWorld", Vec3(cam_world)),
				("iAlbedo", hex_to_rgba(0xD4AF3700).xyz()),
				("iLights", &lights),
				("iMetallicity", metallicity),
				("iRoughness", roughness),
				("iExposure", 1.),
				("iMaxLod", skybox.mip_levels)
			);
			loading_in_progress = !demo_mesh.draw(&model_name);
		}
		{
			let s = skybox.specular.Bind(mipmapped);
			let _ = Uniforms!(skybox_shd, ("skybox_tex", &s), ("MVPMat", cam1.VP()), ("iExposure", 1.));
			Skybox::Draw();
		}

		r.Layout(ID!("menu"), |r, (pos, size)| {
			let (button_w, button_h, padding) = (0.18, 0.06, 0.01);
			let button = |n| (pos.sum(size.mul((button_w * f32(n) + padding * f32(n * 2 + 1), padding))), size.mul((button_w, button_h)));
			let pos = pos.sum(size.mul((0, button_h + padding * 2.)));
			let offset = 0.5 - (button_h + padding * 3.);

			r.TextEdit(ID!("text"), pos, size.mul((1, offset)), 0.03);

			r.HyperText(ID!("hypertext"), pos.sum(size.mul((0, offset + padding))), size.mul((1, 0.5)), 0.05, db);

			let (p, s) = button(4);
			exit = r.Button(ID!("exit"), p, s, "Exit");
			show_tooltip(r.hovers_in(p, s) && !exit, p.sum(s.mul(0.5)), "GUI is easy!", r);
		});

		if loading_in_progress {
			r.draw(primitives::Sprite {
				pos: mouse_pos,
				size: (40., 40.).div(window.aspect()).div(window.size()),
				color: (1., 1. - i, 1., 1.),
				tex: spinner.frame(i),
			});
		}

		let mut events = window.poll_events();
		events.iter().rev().find_map(|e| map_variant!(&MouseMove { at, .. } = e => mouse_pos = at));
		r.sync_clipboard(&mut window);
		renderer = r.unlock(&mut window, &mut events);
		window.swap();

		for e in events {
			match e {
				Keyboard { key, state } if Key::Escape == key && state.pressed() => exit = true,
				Scroll { at, .. } => {
					magnification = (magnification + 0.01 * at.y()).clamp(0.01, 1.);
					rotation += 0.2 * at.x()
				}
				_ => (), //println!("{e:?}"),
			}
		}

		if exit {
			return;
		}
		use event::{Event::*, *};
	}
}

fn timeout(active: bool) -> bool {
	unsafe {
		static mut TIME: usize = 0;
		TIME += 1;
		if !active {
			TIME = 0
		}
		TIME > 60
	}
}

#[derive(Default)]
struct DeferredMesh {
	handle: Option<Task<(String, Res<Model>)>>,
	mesh: Option<(String, Box<dyn AnyMesh>)>,
}
impl DeferredMesh {
	fn draw(&mut self, name: &str) -> bool {
		let Self { handle, mesh } = self;
		if let Some((n, m)) = mesh {
			m.Draw();
			if n == name {
				return true;
			}
		}

		if handle.is_none() {
			*mesh = Some(("".into(), Box(Mesh::make_sphere(0.1, 8))));
			*handle = Some(task::spawn({
				let name = name.to_string();
				async move { (name.clone(), Model::new_cached(&name)) }
			}));
		}

		let handle = handle.as_mut().valid();
		let mut ready = task::block_on(async move { task::poll_once(handle).await });

		if ready.is_some() {
			self.handle = None;
			let (n, m) = ready.take().valid();
			let m: Box<dyn AnyMesh> = m.map_or_else(
				|e| {
					WARN!("Failed to load {n:?}, {e}");
					Mesh::make_sphere(0.1, 8).to_trait()
				},
				|m| Mesh::from(m).to_trait(),
			);
			*mesh = Some((n, m));
		}
		false
	}
}
