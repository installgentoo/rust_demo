#![warn(clippy::all)]
#![allow(clippy::many_single_char_names)]

use grafix_toolbox::{gui::*, *};
use uses::{asyn::*, math::*, *};
use GL::{atlas::*, font::*, mesh::*, pbrt::*, window::*, *};

fn main() {
	LOGGER!(logging::Term, INFO);

	ShaderManager::LoadSources("shd_pbrt.glsl");

	let mut window = TIMER!(window, {
		let win = EXPECT!(Window::get((50, 50, 1700, 900), "Engine"));
		GLEnable!(DEPTH_TEST, BLEND, MULTISAMPLE, TEXTURE_CUBE_MAP_SEAMLESS, DEPTH_WRITEMASK);
		GLDisable!(CULL_FACE);
		GL::BlendFunc::Set((gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA));
		GL::DepthFunc::Set(gl::LESS);
		win
	});

	let font = {
		let alphabet = (9..=9).chain(32_u8..127).map(|n| n as char).collect::<String>() + "ёйцукенгшщзхъфывапролджэячсмитьбюЁЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ";
		Font::new_cached("UbuntuMono-R", alphabet)
	};

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
	let mut spinner = Animation::from_file("spinner", &atlas);

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

	let large_text: String = (0..100)
		.map(|_| "Functional textbox is capable of showing millions of lines with negligeble cpu usage!(so long as we're not editing, but that can be solved functionally as well)\nBiggest issue is collecting the text to display, really - you can set large text range to 1000000 in release.\n\nUpper bar allows to drag layout around, pip on the bottom right resizes it.\n\n\n")
		.collect();

	Layout::storage(ID!("menu")).pos = (-10., -10.);
	Layout::storage(ID!("menu")).size = (1.5, 2.);
	TextEdit::storage(ID!("text")).text = large_text.into();
	Slider::storage(ID!("metallicity")).pip_pos = 0.885;
	Slider::storage(ID!("roughness")).pip_pos = 0.286;
	Slider::storage(ID!("rotation")).pip_pos = 0.1;

	let show_tooltip = |hovered, p: Vec2, msg: &str, r: &mut RenderLock| {
		if timeout(hovered) {
			let size = (0.03, 0.1).mul((msg.len(), 1));
			r.clip(p, size);
			r.Label(ID!("Tooltip"), p.sum(size.mul(0.05)), size, msg);
			r.unclip();
		}
	};

	let mut exit = false;
	let mut mouse_pos = (0., 0.);
	let mut options: Vec<_> = ["stanford_dragon", "dr", "tyra", "not a model"].iter().map(|s| s.to_string()).collect();
	let mut demo_mesh = DeferredMesh::default();
	let mut loading_in_progress;
	let mut magnification = 0.3;
	let mut rotation = 0_f32;

	for i in (0..100).chain((0..100).step_by(1).rev()).map(|i| f32(i) / 100.).cycle() {
		use glm::{vec3 as v3, vec4 as v4, Mat4 as m4};
		let a = window.aspect();
		let mut cam1 = Camera::new(
			glm::perspective(a.y() / a.x(), 70_f32.to_radians(), 0.1, 10.),
			glm::look_at(&v3(0., 0., -5.), &v3(0., 0., 0.), &v3(0., 1., 0.)),
		);

		let mut r = renderer.lock();

		{
			window.bind();
			window.clear((0, 1));

			let metallicity = 0.995 * r.Slider(ID!("metallicity"), (0.3, -0.88), (1., 0.05), 0.05);
			let roughness = (0.05 + r.Slider(ID!("roughness"), (0.3, -0.94), (1., 0.05), 0.05)) / 1.05;
			r.Label(ID!("metal_v"), (1.31, -0.88), (0.3, 0.05), &format!("Metal: {:.3}", metallicity));
			r.Label(ID!("rough_v"), (1.31, -0.94), (0.3, 0.05), &format!("Rough: {:.3}", roughness));
			let model_name = Selector::storage(ID!("model")).choice;
			let model_name = options[model_name].clone();
			r.Selector(ID!("model"), (1.31, -0.82), (0.6, 0.1), &mut options[..]);

			let t = 1. - i;

			let m = magnification;
			let model = &glm::translate(
				&glm::rotate(&glm::scale(&glm::identity(), &v3(m, m, m)), 90_f32.to_radians(), &v3(-1., 0., 0.)),
				&v3(0., 0., -0.3),
			);
			let (c, s, cam_world) = {
				rotation += r.Slider(ID!("rotation"), (0.3, -1.), (1., 0.05), 0.05) * 0.01;
				let (c, s) = (rotation.cos(), rotation.sin()).mul(0.3);
				let t = t.to_radians() * 90.;
				(t.cos(), t.sin(), v3(s, 0., c))
			};

			cam1.setView(glm::look_at(&cam_world, &v3(0., 0., 0.), &v3(0., 1., 0.)));
			let view = m4::to(cam1.V());
			let (irr, spec, lut) = (skybox.irradiance.Bind(sampl), skybox.specular.Bind(mipmapped), brdf_lut.Bind(sampl));
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
				(
					"iLightPos",
					&[
						Vec4(view * v4(6. * c, 6. * s, 0., 1.)).xyz(),
						Vec4(view * v4(2. * c, 0., 2. * s, 1.)).xyz(),
						Vec4(view * v4(c, s, -2., 1.)).xyz(),
						Vec4(view * v4(-1.5 * s, 1. * c, -2., 1.)).xyz()
					][..]
				),
				("iLightColor", &[(1., 0., 0., 2.), (0., 0., 1., 2.), (1., 0., 1., 2.), (0., 1., 0., 2.)][..]),
				("iAlbedo", hex_to_rgba(0xD4AF3700).xyz()),
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

			r.TextEdit(
				ID!("text"),
				pos.sum(size.mul((0, button_h + padding * 2.))),
				size.mul((1, 1. - (button_h + padding * 3.))),
				0.05,
			);

			let (p, s) = button(4);
			exit = r.Button(ID!("exit"), p, s, "Exit");
			show_tooltip(r.hovers_in((p, p.sum(s))) && !exit, p.sum(s.mul(0.5)), "GUI is easy!", r);
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
		events.iter().rev().find_map(|e| map_enum!(MouseMove { at, .. } = e => mouse_pos = *at));
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
				_ => (), //println!("{:?}", e),
			}
		}

		if exit {
			return;
		}
		use events::{Event::*, *};
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
			let name = name.to_string();
			*mesh = Some(("".into(), Box::new(Mesh::make_sphere(0.1, 8))));
			*handle = Some(task::spawn(async move { (name.clone(), Model::new_cached(&name)) }));
		}

		let handle = handle.as_mut().unwrap();
		let mut ready = task::block_on(async move { task::poll_once(handle).await });

		if ready.is_some() {
			self.handle = None;
			let (n, m) = ready.take().unwrap();
			let m: Box<dyn AnyMesh> = m.map_or_else(
				|e| {
					WARN!("Failed to load '{}', {}", n, e);
					Mesh::make_sphere(0.1, 8).to_trait()
				},
				|m| Mesh::from(m).to_trait(),
			);
			*mesh = Some((n, m));
		}
		false
	}
}
