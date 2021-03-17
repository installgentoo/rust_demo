#![warn(clippy::all)]

use GL::{atlas::*, font::*, pbrt::*, window::*, *};
use grafix_toolbox::{gui::*, lazy::*, lib::*, math::la::*, math::*, *};

fn main() {
	LOGGER!(logger::Term, INFO);

	let mut window = TIMER!(window, {
		let mut win = Window((50, 50, 1600, 900), "Engine");
		ShaderManager::Initialize(&mut win);
		GL::BlendFunc::Set((gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA));
		GL::DepthFunc::Set(gl::LESS);
		GLEnable!(DEPTH_TEST, TEXTURE_CUBE_MAP_SEAMLESS);
		win
	});

	ShaderManager::Load("shd_pbrt.glsl");

	let atlas = TexAtlas::<RGBA>::new();

	let font = (9..=10)
		.chain(32_u8..127)
		.map(|n| n as char)
		.chain("ёйцукенгшщзхъфывапролджэячсмитьбюЁЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ".chars())
		.collect::<String>()
		.pipe(|alphabet| Font::new_cached("UbuntuMono-R", alphabet))
		.pipe(Arc);

	#[cfg(debug_assertions)]
	let large_repeats = 10;
	#[cfg(not(debug_assertions))]
	let large_repeats = usize(2e5);

	let large_text = "Functional textbox is capable of showing millions of lines with negligeble cpu usage!(so long as we're not editing, but that can be solved functionally as well)\nBiggest issue is collecting the text to display, really - you can check out million line text by launching in release.\n\nUpper bar allows to drag layout around, pip on the bottom right resizes it.\n\nВстает заря во мгле холодной; На нивах шум работ умолк; С своей волчихою голодной; Выходит на дорогу волк\n\n".repeat(large_repeats);

	let popups_text = "With my approach, you can implement context popups in your ui, in under 250 lines of code. I mean, something like this takes a very complex toolkit and A LOT of code, traditionally. This is the power of functional architecture. The implementation also has ample space for memoizations, in case *fast* isn't fast enough(Observe how textedit has async text parsing to not lag ui on millions of lines, and how simple that was to implement).\n\n\n".repeat(3);

	let db = &[
		(
			"context popup",
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
		(
			"memoization",
			"In perhaps 99% of cases you can just save the result of a function\nin cache var and not think about particulars of your program's memory.\nWhich removes a lot of coupling, which enables easier modification\nonce you need to add some new functionality",
		),
	].pipe(|db| HyperDB::new(&font, db));

	let mut gui = Renderer::new(
		Theme {
		easing: 10.,
		bg: (0.2, 0.2, 0.2, 0.7),
		bg_focus: hex_to_rgba(0x596475A0),
		fg: hex_to_rgba(0x626975FF),
		fg_focus: hex_to_rgba(0x461E5CCF),
		highlight: (0.9, 0.4, 0.1, 1.),
		text: (1., 0.9, 0.9, 0.9),
			text_focus: Vec4(1),
		text_highlight: (0.2, 0.2, 0.2, 1.),
		font,
			font_size: 0.08,
		},
		&window,
	);

	let (mut skybox_shd, mut render_shd) = (("vs_skybox", "ps_skybox"), ("vs_material_based_render", "ps_material_based_render")).map(|n| Shader::new(n).fail());

	let (spinner, skybox, brdf_lut, sampl, mipmapped) = (
		Animation::from_file("spinner", &atlas).fail(),
		Environment::new_cached("lythwood_lounge_4k").fail(),
		Environment::lut_cached(),
		&Sampler::linear(),
		&Sampler!(
			(TEXTURE_WRAP_R, CLAMP_TO_EDGE),
			(TEXTURE_WRAP_S, CLAMP_TO_EDGE),
			(TEXTURE_WRAP_T, CLAMP_TO_EDGE),
			(TEXTURE_MIN_FILTER, LINEAR_MIPMAP_LINEAR)
		),
	);

	gui.Layout(ID!("menu")).layout = (Vec2(-10), (1.5.or_val(large_repeats == 10, || 3.), 2.)).into();
	gui.HyperText(ID!("hypertext")).text = popups_text.into();
	gui.TextEdit(ID!("text")).text = large_text.into();
	gui.Slider(ID!("metallicity")).pip_pos = 0.885;
	gui.Slider(ID!("roughness")).pip_pos = 0.117;
	gui.SliderNum(ID!("rotation")).value = 0.1;

	let tooltip = |hovered, s: Surf, msg: &str, r: &mut RenderLock| {
		if timeout(hovered) {
			let s = s.x_self(0.9).y_self(0.75).size((0.03, 0.1).mul((msg.len(), 1)));
			r.unclipped(|r| r.Label(ID!("Tooltip")).draw(s, msg));
		}
	};

	let mut loading_in_progress;
	let (mut exit, mut mouse_pos, mut rotation, mut cam1, mut demo_mesh) = Def::<(_, _, f32, FocusCam, DeferredMesh)>();
	let (mut magnification, mut options, mut lights) = (
		0.25,
		["stanford_dragon", "dr", "tyra", "not a model"].map(String::from),
		UniformArr::new((vec![0.; 4 + 4 * 20 * 2], gl::DYNAMIC_STORAGE_BIT | gl::MAP_WRITE_BIT)),
	);

	for i in (0..100).chain((0..100).step_by(1).rev()).map(|i| f32(i) / 100.).cycle() {
		let mut r = gui.lock();

		{
			cam1.set_proj(&window, (70., 2.));

			window.bind();
			window.clear((0, 1));

			let s = Surf::new((0.3, -0.88), (1., 0.05));
			let metallicity = 0.995 * r.Slider(ID!("metallicity")).draw(s);
			let roughness = 0.05 + r.Slider(ID!("roughness")).draw(s.y(-0.06)) * 0.95;
			let sl = s.x(1.01).size((0.3, 0.05));
			r.Label(ID!("metal_v")).draw(sl, format!("Metal: {metallicity:.3}"));
			r.Label(ID!("rough_v")).draw(sl.y(-0.06), format!("Rough: {roughness:.3}"));
			let model_name = r.Selector(ID!("model")).draw(sl.y(0.06).size((0.45, 0.1)), &mut options);

			let (v3, v4) = (V3::new, V4::new);

			let model = &(magnification, identity())
				.pipe(|(m, s)| scale(s, v3(m, m, m)))
				.pipe(|s| rotate(s, 90f32.to_radians(), -V3::x_axis()))
				.pipe(|s| translate(s, v3(0., 0., -0.3)));
			rotation += r.SliderNum(ID!("rotation")).draw(s.y(-0.12), (0, 1));

			cam1.set_polar((rotation - 120., -30., 0.3));
			let view = cam1.V();
			{
				let t = 1. - i;
				let (s, c) = (t.to_radians() * 90.).sin_cos();

				let l = lights.array.MapMut(..4 + 4 * 4 * 2).mem();
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
				("iIrradiance", irr),
				("iSpecular", spec),
				("iLut", lut),
				("MVPMat", cam1.MVP(model)),
				("ModelViewMat", cam1.MV(model)),
				("NormalViewMat", cam1.NV(model)),
				("NormalMat", cam1.N(model)),
				("iCameraWorld", cam1.pos()),
				("iAlbedo", hex_to_rgba(0xD4AF3700).xyz()),
				("iLights", lights),
				("iMetallicity", metallicity),
				("iRoughness", roughness),
				("iExposure", 1.),
				("iMaxLod", skybox.mip_levels)
			);
			loading_in_progress = !demo_mesh.draw(model_name);
		}
		{
			let s = skybox.specular.Bind(mipmapped);
			let _ = Uniforms!(skybox_shd, ("iSkybox", s), ("MVPMat", cam1.VP()), ("iExposure", 1.));
			Skybox::Draw();
		}

		r.Layout(ID!("menu")).draw(|r, s @ Surf { size, .. }| {
			let (btn_size, pad) = ((0.2, 0.08), 0.01);
			let btn_scaled = btn_size.mul(size);
			let button = |n| s.size(btn_scaled).x_self(n).xy((pad, pad)).size_sub(Vec2(2).mul(pad));
			let s = s.y(btn_scaled.y()).h_scale((1. - btn_size.y() - pad * 0.5) * 0.5);

			let p = pix_to_size(1., &window);
			r.TextEdit(ID!("text")).draw(s, 8. * p);
			r.HyperText(ID!("hypertext")).draw(s.y_self(1).y(pad), 10. * p, db);

			let sb = button(4.);
			exit = r.Button(ID!("exit")).draw(sb, "Exit");
			tooltip(r.hovers_in(sb) && !exit, sb, "GUI is easy!", r);
		});

		if loading_in_progress {
			r.draw(prim::Sprite {
				pos: mouse_pos,
				size: Vec2(40).mul(window.aspect()).div(window.size()),
				color: (1., 1. - i, 1., 1.),
				tex: spinner.frame(i),
			});
		}

		let mut events = window
			.poll_events()
			.tap(|e| e.iter().rev().find_map(|e| map_variant!(&MouseMove { at, .. } = e => mouse_pos = at)).sink());

		gui = r.tap(|r| r.sync_clipboard(&mut window)).unlock(&mut window, &mut events);
		window.swap();

		for e in events {
			match e {
				Keyboard { key, m } if Key::Escape == key && m.pressed() => exit = true,
				Scroll { at, .. } => {
					magnification = (magnification + 0.01 * at.y()).clamp(0.01, 1.);
					rotation += 0.2 * at.x()
				}
				_ => (), //PRINT!("{e:?}"),
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
	loader: Feed<Option<Model>>,
	mesh: AnyMesh,
	last_name: Astr,
}
impl DeferredMesh {
	fn draw(&mut self, name: &str) -> bool {
		let Self { loader, mesh, last_name } = self;

		mesh.Draw();

		if name != &**last_name {
			let name: Astr = name.into();
			(*mesh, *last_name, *loader) = (
				Def(),
				name.clone(),
				Feed::lazy(async move || Model::new_cached(&name).map_err(|e| WARN!("Cannot load {name:?}, {e}")).ok()),
			);
		}

		let Some(model) = loader.try_lock() else { return false };

		if let Some(model) = model.try_take() {
			*mesh = model.into();
		}

		true
	}
}
